from __future__ import annotations

import argparse
import os
import re
import urllib.request
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

from .common import ARTIFACTS, ensure_dirs, protocol, sha256_bytes, sha256_file, verify_protocol_hash, write_json, write_jsonl


def normalize_text(value: str) -> str:
    return re.sub(r"\s+", " ", value.strip().lower())


def text_sha256(value: str) -> str:
    return sha256_bytes(normalize_text(value).encode("utf-8"))


def _download(url: str, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists() and destination.stat().st_size > 0:
        return
    temporary = destination.with_suffix(destination.suffix + ".download")
    with urllib.request.urlopen(url, timeout=120) as response, temporary.open("wb") as stream:
        while chunk := response.read(1024 * 1024):
            stream.write(chunk)
    temporary.replace(destination)


def _records(frame: pd.DataFrame, source_split: str) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for source_index, row in frame.iterrows():
        text = str(row["text"])
        digest = text_sha256(text)
        rows.append(
            {
                "object_id": f"ag_news:{source_split}:{int(source_index):06d}:{digest[:12]}",
                "source_split": source_split,
                "source_index": int(source_index),
                "text": text,
                "normalized_text_sha256": digest,
                "label": int(row["label"]),
            }
        )
    return rows


def prepare() -> dict[str, object]:
    frozen = protocol()
    cfg = frozen["modern_contour"]
    ensure_dirs()
    revision = cfg["dataset"]["revision"]
    base = f"https://huggingface.co/datasets/{cfg['dataset']['id']}/resolve/{revision}/data"
    raw = ARTIFACTS / "raw"
    paths = {
        "train": raw / "train-00000-of-00001.parquet",
        "test": raw / "test-00000-of-00001.parquet",
    }
    for split, path in paths.items():
        _download(f"{base}/{path.name}", path)

    original_train = pd.read_parquet(paths["train"])
    original_test = pd.read_parquet(paths["test"])
    if len(original_train) != cfg["dataset"]["source_train_size"] or len(original_test) != cfg["dataset"]["source_test_size"]:
        raise RuntimeError("upstream AG News sizes differ from the frozen protocol")

    # The pinned upstream revision contains repeated training texts and ten
    # exact train/test duplicates. Keep the official test intact and exclude
    # all conflicting/repeated training rows before any model run.
    train_normalized = original_train["text"].astype(str).map(normalize_text)
    test_normalized = original_test["text"].astype(str).map(normalize_text)
    test_values = set(test_normalized)
    shared_with_test = train_normalized.isin(test_values)
    duplicate_training = train_normalized.duplicated(keep="first")
    excluded = original_train.loc[shared_with_test | duplicate_training].copy()
    filtered_train = original_train.loc[~(shared_with_test | duplicate_training)].copy()
    deviation = {
        "status": "registered_before_model_execution",
        "reason": "pinned_upstream_revision_contains_exact_normalized_text_duplicates",
        "original_protocol_train_size": cfg["split"]["train"],
        "actual_train_pool_after_exclusion": len(filtered_train),
        "excluded_training_rows": len(excluded),
        "excluded_train_test_matches": int(shared_with_test.sum()),
        "excluded_repeated_training_rows": int(duplicate_training.sum()),
        "official_test_preserved": True,
        "thresholds_models_and_metrics_changed": False,
    }
    write_json(ARTIFACTS / "manifests" / "protocol_deviation_duplicate_rows.json", deviation)

    all_train_indices = filtered_train.index.to_numpy()
    train_indices, validation_indices = train_test_split(
        all_train_indices,
        test_size=cfg["split"]["validation"],
        random_state=frozen["statistics"]["seeds"][0],
        stratify=filtered_train["label"].to_numpy(),
    )
    train_rows = _records(filtered_train.loc[sorted(train_indices)], "train")
    validation_rows = _records(filtered_train.loc[sorted(validation_indices)], "validation")
    test_rows_with_labels = _records(original_test, "sealed_test")
    test_inputs = [{key: value for key, value in row.items() if key != "label"} for row in test_rows_with_labels]
    test_labels = {str(row["object_id"]): int(row["label"]) for row in test_rows_with_labels}

    processed = ARTIFACTS / "processed"
    private = ARTIFACTS / "private"
    write_jsonl(processed / "train.jsonl", train_rows)
    write_jsonl(processed / "validation.jsonl", validation_rows)
    write_jsonl(processed / "sealed_test_inputs.jsonl", test_inputs)
    label_path = private / "sealed_test_labels.json"
    write_json(label_path, {"labels": test_labels})
    os.chmod(label_path, 0o600)

    split_sets = {
        "train": {str(row["object_id"]) for row in train_rows},
        "validation": {str(row["object_id"]) for row in validation_rows},
        "sealed_test": {str(row["object_id"]) for row in test_inputs},
    }
    content_sets = {
        "train": {str(row["normalized_text_sha256"]) for row in train_rows},
        "validation": {str(row["normalized_text_sha256"]) for row in validation_rows},
        "sealed_test": {str(row["normalized_text_sha256"]) for row in test_inputs},
    }
    overlaps = {}
    for left, right in (("train", "validation"), ("train", "sealed_test"), ("validation", "sealed_test")):
        overlaps[f"{left}__{right}"] = {
            "identity": len(split_sets[left] & split_sets[right]),
            "normalized_text": len(content_sets[left] & content_sets[right]),
        }
    audit = {
        "protocol_sha256": verify_protocol_hash(),
        "counts": {name: len(values) for name, values in split_sets.items()},
        "overlaps": overlaps,
        "test_labels_in_public_test_rows": False,
        "passed": all(item["identity"] == 0 and item["normalized_text"] == 0 for item in overlaps.values()),
    }
    write_json(ARTIFACTS / "leakage_audit.json", audit)
    if not audit["passed"]:
        raise RuntimeError(f"split leakage detected: {overlaps}")

    manifest = {
        "dataset": cfg["dataset"],
        "raw_files": {name: {"path": str(path.relative_to(ARTIFACTS)), "sha256": sha256_file(path), "bytes": path.stat().st_size} for name, path in paths.items()},
        "processed_files": {
            name: {"path": str(path.relative_to(ARTIFACTS)), "sha256": sha256_file(path), "rows": len(split_sets[name])}
            for name, path in {
                "train": processed / "train.jsonl",
                "validation": processed / "validation.jsonl",
                "sealed_test": processed / "sealed_test_inputs.jsonl",
            }.items()
        },
        "private_label_vault": {"path": str(label_path.relative_to(ARTIFACTS)), "sha256": sha256_file(label_path), "release_excluded": True},
        "split_id_sha256": {name: sha256_bytes("\n".join(sorted(values)).encode("utf-8")) for name, values in split_sets.items()},
        "license": {
            "upstream_status": cfg["dataset"]["license_status"],
            "release_redistribution": cfg["dataset"]["redistribution"],
            "source_url": cfg["dataset"]["source"],
        },
        "protocol_deviation": deviation,
    }
    write_json(ARTIFACTS / "manifests" / "dataset_manifest.json", manifest)
    return {"counts": audit["counts"], "audit_passed": audit["passed"], "manifest": str(ARTIFACTS / "manifests" / "dataset_manifest.json")}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.parse_args()
    result = prepare()
    print(f"PASS: prepared AG News {result['counts']} leakage={result['audit_passed']}")


if __name__ == "__main__":
    main()
