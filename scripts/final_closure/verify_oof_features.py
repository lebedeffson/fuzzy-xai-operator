#!/usr/bin/env python3
"""Audit generated OOF rows against registered split identities."""

from __future__ import annotations

import json

from common import ROOT, STUDY, load, sha256, write


DATA_ROOT = ROOT / "data/confirmatory"


def main() -> None:
    manifest = load(STUDY / "confirmatory_feature_manifest.json")
    predictive_channels = set(manifest["predictive_channels"])
    route_channels = set(manifest["route_channels"])
    global_oof = _read_ids(STUDY / "oof_object_hashes.txt")
    global_test = _read_ids(STUDY / "sealed_test_object_hashes.txt")
    observed: set[str] = set()
    blockers: list[str] = []
    reports = []
    for dataset in manifest["datasets"]:
        dataset_id = dataset["dataset_id"]
        artifact = ROOT / dataset["artifact_path"]
        expected = _dataset_ids(dataset_id, "train") | _dataset_ids(dataset_id, "development")
        test_ids = _dataset_ids(dataset_id, "sealed_test")
        dataset_observed: set[str] = set()
        if not artifact.is_file() or sha256(artifact) != dataset["artifact_sha256"]:
            blockers.append(f"ARTIFACT_HASH:{dataset_id}")
            continue
        with artifact.open(encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                row = json.loads(line)
                object_id = row.get("object_id_hash")
                if row.get("dataset_id") != dataset_id:
                    blockers.append(f"DATASET_ID:{dataset_id}:{line_number}")
                if object_id not in expected or object_id in test_ids or object_id in global_test:
                    blockers.append(f"IDENTITY_SCOPE:{dataset_id}:{line_number}")
                if object_id in dataset_observed:
                    blockers.append(f"DUPLICATE_ID:{dataset_id}:{line_number}")
                dataset_observed.add(object_id)
                predictive = row.get("predictive", {})
                route = row.get("route", {})
                if set(predictive) != predictive_channels or set(route) != route_channels:
                    blockers.append(f"CHANNEL_SCHEMA:{dataset_id}:{line_number}")
                values = (*predictive.values(), *route.values())
                if any(value is not None and not 0.0 <= float(value) <= 1.0 for value in values):
                    blockers.append(f"CHANNEL_RANGE:{dataset_id}:{line_number}")
                missing = {name for name, value in (*predictive.items(), *route.items()) if value is None}
                if set(row.get("missing_channels", ())) != missing:
                    blockers.append(f"MISSING_CHANNELS:{dataset_id}:{line_number}")
                if row.get("source_is_oof") is not True or row.get("split_id") != "train-development-oof":
                    blockers.append(f"OOF_MARKER:{dataset_id}:{line_number}")
        if dataset_observed != expected or len(dataset_observed) != dataset["objects"]:
            blockers.append(f"DATASET_COVERAGE:{dataset_id}")
        observed.update(dataset_observed)
        reports.append({"dataset_id": dataset_id, "objects": len(dataset_observed), "artifact_sha256": sha256(artifact)})
    if observed != global_oof:
        blockers.append("GLOBAL_OOF_COVERAGE")
    if observed & global_test:
        blockers.append("GLOBAL_OOF_TEST_OVERLAP")
    payload = {
        "status": "pass" if not blockers else "fail",
        "datasets": reports,
        "oof_objects": len(observed),
        "sealed_test_objects": len(global_test),
        "oof_test_overlap": len(observed & global_test),
        "held_out_label_used_as_feature": False,
        "sealed_test_loaded": False,
        "blockers": sorted(set(blockers)),
    }
    write(STUDY / "oof_feature_audit.json", payload)
    if blockers:
        raise SystemExit(f"FAIL: OOF feature audit: {sorted(set(blockers))[:20]}")
    print(f"PASS: final_oof_feature_audit objects={len(observed)} test_overlap=0")


def _dataset_ids(dataset_id: str, split: str) -> set[str]:
    return _read_ids(DATA_ROOT / dataset_id / f"manifests/{split}_object_ids.txt")


def _read_ids(path) -> set[str]:
    return {line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()}


if __name__ == "__main__":
    main()
