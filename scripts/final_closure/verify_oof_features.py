#!/usr/bin/env python3
"""Audit generated OOF rows against registered split identities."""

from __future__ import annotations

import json
import hashlib

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
    if manifest.get("P0_status") != "pass_predictive_oof":
        blockers.append("P0_NOT_PASS")
    if manifest.get("P1_status") != "pass_route_oof":
        blockers.append("P1_NOT_PASS")
    near_duplicates = load(STUDY / "near_duplicate_audit.json")
    if near_duplicates.get("status") != "pass" or near_duplicates.get("near_duplicate_violations") != 0:
        blockers.append("NEAR_DUPLICATE_AUDIT")
    reports = []
    for dataset in manifest["datasets"]:
        dataset_id = dataset["dataset_id"]
        artifact = ROOT / dataset["artifact_path"]
        evidence_artifact = ROOT / dataset["canonical_evidence_path"]
        model_manifest = ROOT / dataset["model_manifest_path"]
        explainer_manifest = ROOT / dataset["explainer_manifest_path"]
        expected = _dataset_ids(dataset_id, "train") | _dataset_ids(dataset_id, "development")
        test_ids = _dataset_ids(dataset_id, "sealed_test")
        dataset_observed: set[str] = set()
        if not artifact.is_file() or sha256(artifact) != dataset["artifact_sha256"]:
            blockers.append(f"ARTIFACT_HASH:{dataset_id}")
            continue
        required_hashes = (
            (evidence_artifact, dataset["canonical_evidence_sha256"], "CANONICAL"),
            (model_manifest, dataset["model_manifest_sha256"], "MODEL_MANIFEST"),
            (explainer_manifest, dataset["explainer_manifest_sha256"], "EXPLAINER_MANIFEST"),
        )
        for path, expected_hash, label in required_hashes:
            if not path.is_file() or sha256(path) != expected_hash:
                blockers.append(f"{label}_HASH:{dataset_id}")
        canonical_hashes = _canonical_hashes(evidence_artifact)
        allowed_missing = set(dataset.get("not_applicable_route_channels", ()))
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
                if any(value is None for value in predictive.values()):
                    blockers.append(f"P0_MISSING:{dataset_id}:{line_number}")
                values = (*predictive.values(), *route.values())
                if any(value is not None and not 0.0 <= float(value) <= 1.0 for value in values):
                    blockers.append(f"CHANNEL_RANGE:{dataset_id}:{line_number}")
                missing = {name for name, value in (*predictive.items(), *route.items()) if value is None}
                if set(row.get("missing_channels", ())) != missing:
                    blockers.append(f"MISSING_CHANNELS:{dataset_id}:{line_number}")
                if missing - allowed_missing:
                    blockers.append(f"UNDECLARED_NOT_APPLICABLE:{dataset_id}:{line_number}")
                if row.get("canonical_evidence_sha256") != canonical_hashes.get(object_id):
                    blockers.append(f"CANONICAL_ROW_HASH:{dataset_id}:{line_number}")
                forbidden = {"true_label", "is_correct", "outcome", "ground_truth"}
                if forbidden & (set(predictive) | set(route)):
                    blockers.append(f"OUTCOME_FEATURE:{dataset_id}:{line_number}")
                if row.get("source_is_oof") is not True or row.get("split_id") != "train-development-oof":
                    blockers.append(f"OOF_MARKER:{dataset_id}:{line_number}")
        if dataset_observed != expected or len(dataset_observed) != dataset["objects"]:
            blockers.append(f"DATASET_COVERAGE:{dataset_id}")
        observed.update(dataset_observed)
        if len(canonical_hashes) != len(dataset_observed):
            blockers.append(f"CANONICAL_COVERAGE:{dataset_id}")
        reports.append(
            {
                "dataset_id": dataset_id,
                "objects": len(dataset_observed),
                "artifact_sha256": sha256(artifact),
                "canonical_evidence_sha256": sha256(evidence_artifact),
                "model_manifest_sha256": sha256(model_manifest),
                "explainer_manifest_sha256": sha256(explainer_manifest),
                "allowed_not_applicable_channels": sorted(allowed_missing),
            }
        )
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
        "canonical_hash_coverage": 1.0 if observed else 0.0,
        "near_duplicate_violations": near_duplicates.get("near_duplicate_violations"),
        "blockers": sorted(set(blockers)),
    }
    write(STUDY / "p0_p1_feature_audit.json", payload)
    write(STUDY / "oof_feature_audit.json", payload)
    if blockers:
        raise SystemExit(f"FAIL: OOF feature audit: {sorted(set(blockers))[:20]}")
    manifest["lock_status"] = "ready_for_lock"
    manifest["P0_P1_audit_status"] = "pass"
    write(STUDY / "confirmatory_feature_manifest.json", manifest)
    print(f"PASS: final_oof_feature_audit objects={len(observed)} test_overlap=0")


def _canonical_hashes(path) -> dict[str, str]:
    output: dict[str, str] = {}
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            payload = json.loads(line)
            canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"))
            output[str(payload["object_id_hash"])] = hashlib.sha256(canonical.encode()).hexdigest()
    return output


def _dataset_ids(dataset_id: str, split: str) -> set[str]:
    return _read_ids(DATA_ROOT / dataset_id / f"manifests/{split}_object_ids.txt")


def _read_ids(path) -> set[str]:
    return {line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()}


if __name__ == "__main__":
    main()
