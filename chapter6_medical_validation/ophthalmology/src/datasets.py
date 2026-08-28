from __future__ import annotations

import csv
import json
import os
from collections.abc import Iterable
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, cast

import numpy as np
import yaml
from sklearn.model_selection import train_test_split

from .artifact_io import relative_inventory, sha256_file, sha256_json, write_json_once


@dataclass(frozen=True)
class DatasetRecord:
    dataset_id: str
    sample_id: str
    image_path: Path
    label: int
    split: str = "unassigned"

    def public_dict(self, data_root: Path) -> dict[str, Any]:
        return {
            "dataset_id": self.dataset_id,
            "sample_id": self.sample_id,
            "image_path": self.image_path.resolve().relative_to(data_root.resolve()).as_posix(),
            "label": self.label,
            "split": self.split,
        }


def load_yaml(path: str | Path) -> dict[str, Any]:
    value = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"configuration must be a mapping: {path}")
    return value


def configured_data_root(explicit: str | Path | None = None) -> Path:
    value = str(explicit) if explicit is not None else os.environ.get("FUZZYXAI_CH6_DATA_ROOT", "")
    if not value:
        raise FileNotFoundError("FUZZYXAI_CH6_DATA_ROOT is not set; see DATA_ACCESS.md")
    root = Path(value).expanduser()
    if not root.is_dir():
        raise FileNotFoundError(f"configured CH6 data root does not exist: {root}")
    return root


def _read_csv(path: Path) -> list[dict[str, str]]:
    if not path.is_file():
        raise FileNotFoundError(path)
    with path.open(newline="", encoding="utf-8-sig") as stream:
        return [dict(row) for row in csv.DictReader(stream)]


def load_aptos_records(data_root: Path, config: dict[str, Any]) -> list[DatasetRecord]:
    csv_path = data_root / str(config["relative_csv"])
    images = data_root / str(config["relative_images"])
    rows = _read_csv(csv_path)
    expected = int(config["expected_count"])
    if len(rows) != expected:
        raise ValueError(f"APTOS labeled row count {len(rows)} != registered {expected}")
    labels_allowed = {int(value) for value in config["labels"]}
    id_column, label_column = str(config["id_column"]), str(config["label_column"])
    records: list[DatasetRecord] = []
    seen: set[str] = set()
    for row in rows:
        sample_id = str(row[id_column]).strip()
        label = int(row[label_column])
        if not sample_id or sample_id in seen:
            raise ValueError(f"empty or duplicate APTOS sample id: {sample_id!r}")
        if label not in labels_allowed:
            raise ValueError(f"APTOS label outside 0..4: {label}")
        path = images / f"{sample_id}{config.get('image_extension', '.png')}"
        if not path.is_file():
            raise FileNotFoundError(path)
        seen.add(sample_id)
        records.append(DatasetRecord(str(config["dataset_id"]), sample_id, path, label))
    return records


def deterministic_aptos_split(records: list[DatasetRecord], config: dict[str, Any]) -> dict[str, list[DatasetRecord]]:
    split = dict(config["split"])
    counts = {str(key): int(value) for key, value in dict(split["counts"]).items()}
    if sum(counts.values()) != len(records):
        raise ValueError("registered APTOS split counts do not sum to the available labeled records")
    ids = np.asarray([record.sample_id for record in records])
    labels = np.asarray([record.label for record in records])
    indices = np.arange(len(records))
    train_idx, remainder_idx = train_test_split(
        indices,
        train_size=counts["train"],
        random_state=int(split["seed"]),
        stratify=labels,
    )
    val_idx, test_idx = train_test_split(
        remainder_idx,
        train_size=counts["validation"],
        random_state=int(split["seed"]) + 1,
        stratify=labels[remainder_idx],
    )
    del ids

    def assigned(indexes: Iterable[int], name: str) -> list[DatasetRecord]:
        return [DatasetRecord(**{**asdict(records[int(index)]), "split": name}) for index in sorted(indexes, key=lambda i: records[int(i)].sample_id)]

    result = {
        "train": assigned(train_idx, "train"),
        "validation": assigned(val_idx, "validation"),
        "internal_test": assigned(test_idx, "internal_test"),
    }
    all_paths = [record.image_path.resolve() for values in result.values() for record in values]
    if len(all_paths) != len(set(all_paths)):
        raise AssertionError("APTOS image path appears in more than one split")
    return result


def freeze_aptos_split(data_root: Path, config: dict[str, Any], output: Path) -> dict[str, Any]:
    records = load_aptos_records(data_root, config)
    splits = deterministic_aptos_split(records, config)
    payload = {
        "schema_version": "1.0",
        "dataset_id": config["dataset_id"],
        "seed": int(config["split"]["seed"]),
        "method": config["split"]["method"],
        "patient_identifier_status": config["split"]["patient_identifier_status"],
        "source_csv_sha256": sha256_file(data_root / config["relative_csv"]),
        "counts": {name: len(values) for name, values in splits.items()},
        "records": {
            name: [record.public_dict(data_root) for record in values]
            for name, values in splits.items()
        },
    }
    payload["manifest_sha256"] = sha256_json(payload)
    write_json_once(output, payload)
    return payload


def _detect_columns(rows: list[dict[str, str]]) -> tuple[str, str]:
    if not rows:
        raise ValueError("IDRiD label CSV is empty")
    keys = list(rows[0])
    id_key = next((key for key in keys if key.lower() in {"image name", "image_name", "id", "id_code"}), keys[0])
    label_key = next((key for key in keys if "retinopathy grade" in key.lower() or key.lower() in {"diagnosis", "dr_grade", "label"}), "")
    if not label_key:
        raise ValueError(f"cannot identify IDRiD DR-grade column in {keys}")
    return id_key, label_key


def load_idrid_grading_split(data_root: Path, config: dict[str, Any], split: str) -> list[DatasetRecord]:
    grading = dict(config["grading"])
    if split not in {"train", "test"}:
        raise ValueError("IDRiD grading split must be train or test")
    rows = _read_csv(data_root / str(grading[f"{split}_csv"]))
    expected = int(grading["official_counts"][split])
    if len(rows) != expected:
        raise ValueError(f"IDRiD {split} count {len(rows)} != official {expected}")
    id_key, label_key = _detect_columns(rows)
    image_root = data_root / str(grading[f"{split}_images"])
    records: list[DatasetRecord] = []
    for row in rows:
        sample_id = str(row[id_key]).strip()
        label = int(float(row[label_key]))
        if label not in {0, 1, 2, 3, 4}:
            raise ValueError(f"IDRiD label outside 0..4: {label}")
        candidates = [image_root / sample_id, image_root / f"{sample_id}.jpg", image_root / f"{sample_id}.png"]
        image_path = next((path for path in candidates if path.is_file()), candidates[1])
        if not image_path.is_file():
            raise FileNotFoundError(image_path)
        records.append(DatasetRecord(str(config["dataset_id"]), sample_id, image_path, label, f"official_{split}"))
    return records


def build_dataset_manifest(data_root: Path, records: list[DatasetRecord], output: Path, *, source: str) -> dict[str, Any]:
    paths = [record.image_path for record in records]
    payload = {
        "schema_version": "1.0",
        "source": source,
        "count": len(records),
        "labels": sorted({record.label for record in records}),
        "inventory": relative_inventory(data_root, paths),
    }
    payload["manifest_sha256"] = sha256_json(payload)
    write_json_once(output, payload)
    return payload


def assert_no_lesion_masks_in_classifier_inputs(paths: Iterable[Path]) -> None:
    forbidden = {"mask", "masks", "segmentation", "microaneurysm", "haemorrhage", "hemorrhage", "exudate", "optic_disc"}
    for path in paths:
        lowered = {part.lower() for part in path.parts}
        if any(any(term in part for term in forbidden) for part in lowered):
            raise ValueError(f"lesion/segmentation artifact cannot enter classifier input: {path}")


def read_frozen_split(path: str | Path) -> dict[str, Any]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise TypeError("frozen split manifest must be a mapping")
    expected = payload.pop("manifest_sha256")
    actual = sha256_json(payload)
    payload["manifest_sha256"] = expected
    if actual != expected:
        raise ValueError("frozen split manifest checksum mismatch")
    return cast(dict[str, Any], payload)
