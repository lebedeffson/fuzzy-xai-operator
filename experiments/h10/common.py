from __future__ import annotations

import csv
import hashlib
import json
import os
import subprocess
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import yaml


ROOT = Path(__file__).resolve().parents[2]
DATA_ROOT = ROOT / "data" / "h10_v19"
ARTIFACT_ROOT = ROOT / "artifacts" / "h10_v19"
PRIVATE_ROOT = ROOT / ".h10_v19_private"
IDENTITY_ANCHORS = ROOT / "data_seed" / "v19_identity_anchors.json"


@dataclass(frozen=True)
class DatasetSpec:
    dataset_id: str
    modality: str
    license: str
    source: str


DATASETS = (
    DatasetSpec("uci_raisin", "tabular", "CC BY 4.0", "UCI Machine Learning Repository"),
    DatasetSpec("uci_sentiment_labelled_sentences", "text", "CC BY 4.0", "UCI Machine Learning Repository"),
    DatasetSpec("ucr_forda", "time_series", "UCR Archive research dataset", "UCR Time Series Classification Archive"),
)


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def load_yaml(path: Path) -> dict[str, Any]:
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def prepare_datasets() -> list[dict[str, Any]]:
    anchors = read_json(IDENTITY_ANCHORS)
    specs = {item.dataset_id: item for item in DATASETS}
    records: list[dict[str, Any]] = []
    for dataset_id, spec in specs.items():
        identities = sorted({item["object_id"] for item in anchors if item["dataset_id"] == dataset_id})
        split_rows = []
        for object_id in identities:
            bucket = int(sha256_bytes(f"h10-v19:{dataset_id}:{object_id}".encode())[:8], 16) % 100
            split = "train" if bucket < 55 else "development" if bucket < 78 else "sealed_test"
            split_rows.append({"object_id": object_id, "split": split})
        split_path = DATA_ROOT / dataset_id / "manifests" / "split_manifest.json"
        write_json(split_path, split_rows)
        counts = {split: sum(row["split"] == split for row in split_rows) for split in ("train", "development", "sealed_test")}
        records.append(
            {
                **asdict(spec),
                "objects": len(identities),
                "split_counts": counts,
                "identity_anchor_manifest": str(IDENTITY_ANCHORS.relative_to(ROOT)),
                "identity_anchor_sha256": sha256_file(IDENTITY_ANCHORS),
                "split_manifest_sha256": sha256_file(split_path),
                "note": "v19 uses a new salted split and new mutation schedule; predictive labels are not H10 targets",
            }
        )
    write_json(ARTIFACT_ROOT / "data" / "dataset_manifest.json", records)
    return records


def load_identities(dataset_id: str, split: str) -> list[str]:
    rows = read_json(DATA_ROOT / dataset_id / "manifests" / "split_manifest.json")
    return [row["object_id"] for row in rows if row["split"] == split]


def git_commit() -> str:
    return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()


def environment_manifest() -> dict[str, Any]:
    import platform
    import sys

    return {
        "python": sys.version,
        "platform": platform.platform(),
        "processor": platform.processor(),
        "cpu_count": os.cpu_count(),
        "commit": git_commit(),
    }


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    names = fieldnames or sorted({key for row in rows for key in row})
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=names, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
