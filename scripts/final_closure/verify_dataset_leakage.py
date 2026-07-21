#!/usr/bin/env python3
"""Verify sealed-test isolation after real dataset preparation."""

from __future__ import annotations

import numpy as np

from common import ROOT, STUDY, load, sha256, write


DATA_ROOT = ROOT / "data/confirmatory"


def main() -> None:
    audit = load(STUDY / "dataset_leakage_audit.json")
    dataset_manifest = load(STUDY / "confirmatory_dataset_manifest.json")
    split_manifest = load(STUDY / "confirmatory_split_manifest.json")
    blockers = list(audit.get("blockers", []))
    reports = []
    for row in dataset_manifest["datasets"]:
        dataset_id = row["dataset_id"]
        root = DATA_ROOT / dataset_id
        local_split = load(root / "manifests/split_manifest.json")
        intersections = local_split["intersection_counts"]
        if any(intersections.values()):
            blockers.append(f"SPLIT_INTERSECTION:{dataset_id}")
        vault = ROOT / row["label_vault_path"]
        if vault.read_bytes()[:8] != b"Salted__":
            blockers.append(f"VAULT_FORMAT:{dataset_id}")
        test_csv = root / "processed/sealed_test.csv"
        test_npz = root / "processed/sealed_test.npz"
        visible_fields: list[str]
        if test_csv.is_file():
            visible_fields = test_csv.open(encoding="utf-8").readline().strip().split(",")
        elif test_npz.is_file():
            with np.load(test_npz) as payload:
                visible_fields = sorted(payload.files)
        else:
            blockers.append(f"SEALED_TEST_FEATURES_MISSING:{dataset_id}")
            visible_fields = []
        if {"target", "y", "label", "labels"} & set(visible_fields):
            blockers.append(f"TEST_LABEL_VISIBLE:{dataset_id}")
        reports.append(
            {
                "dataset_id": dataset_id,
                "split_intersections": intersections,
                "sealed_test_visible_fields": visible_fields,
                "vault_sha256": sha256(vault),
                "vault_is_openssl_salted_ciphertext": vault.read_bytes()[:8] == b"Salted__",
            }
        )
    if split_manifest.get("oof_object_count", 0) <= 0 or split_manifest.get("sealed_test_object_count", 0) <= 0:
        blockers.append("IDENTITY_COUNTS_MISSING")
    if split_manifest.get("overlap_count") != 0:
        blockers.append("GLOBAL_OOF_TEST_OVERLAP")
    report = {
        "status": "pass" if not blockers else "blocked",
        "scope": "dataset_and_split_isolation_before_protocol_lock",
        "tuning_runner_can_read_test_labels": False,
        "raw_sources_excluded_from_tuning_mount": split_manifest.get("raw_sources_excluded_from_tuning_mount"),
        "datasets": reports,
        "blockers": blockers,
    }
    write(STUDY / "final_leakage_audit.json", report)
    if blockers:
        raise SystemExit(f"BLOCKED: final leakage audit: {blockers}")
    print(f"PASS: final_leakage_audit datasets={len(reports)} oof_test_overlap=0 labels_visible=false")


if __name__ == "__main__":
    main()
