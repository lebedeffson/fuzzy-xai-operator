#!/usr/bin/env python3
"""Validate externally downloaded and encrypted dataset manifests; never fabricate them."""

from __future__ import annotations

from fuzzyxai.final_closure import SealedDataset, audit_registry

from common import ROOT, STUDY, load, sha256, write


def main() -> None:
    supplied = STUDY / "confirmatory_dataset_manifest.input.json"
    split = STUDY / "confirmatory_split_manifest.input.json"
    if not supplied.is_file() or not split.is_file():
        raise SystemExit("BLOCKED: real downloaded dataset and split manifests are required")
    payload, split_payload = load(supplied), load(split)
    registry = load(STUDY / "confirmatory_dataset_registry.json")
    datasets = tuple(SealedDataset(**{key: value for key, value in row.items() if key != "label_vault_path"}) for row in payload["datasets"])
    for row in payload["datasets"]:
        vault = (ROOT / row["label_vault_path"]).resolve()
        if ROOT not in vault.parents or vault.suffix != ".enc":
            raise SystemExit(f"FAIL: label vault must be an encrypted workspace-local artifact for {row['dataset_id']}")
        if not vault.is_file() or sha256(vault) != row["label_vault_sha256"]:
            raise SystemExit(f"FAIL: invalid encrypted label vault for {row['dataset_id']}")
    required_isolation = {
        "tuning_runner_can_read_test_labels": False,
        "test_labels_loaded_by_tuning": False,
        "controller_feature_source": "out_of_fold_train_development_only",
        "test_identity_visibility_during_tuning": "hash_only",
    }
    isolation_blockers = [
        f"ISOLATION_CONTRACT:{key}"
        for key, expected in required_isolation.items()
        if split_payload.get(key) != expected
    ]
    if isolation_blockers:
        raise SystemExit(f"BLOCKED: split isolation contract: {isolation_blockers}")
    oof_hashes = _identity_hashes(split_payload, "oof_object_hashes")
    test_hashes = _identity_hashes(split_payload, "sealed_test_object_hashes")
    audit = audit_registry(
        datasets,
        formative_dataset_ids=set(registry["known_formative_dataset_ids"]),
        formative_hashes=set(registry["known_formative_hashes"]),
        oof_object_hashes=oof_hashes,
        sealed_test_object_hashes=test_hashes,
        tuning_runner_can_read_test_labels=bool(split_payload["tuning_runner_can_read_test_labels"]),
    )
    audit["isolation_contract"] = required_isolation
    audit["identity_source"] = "internally_built_confirmatory_dataset_registry"
    write(STUDY / "dataset_leakage_audit.json", audit)
    if audit["status"] != "pass":
        raise SystemExit(f"BLOCKED: dataset leakage audit: {audit['blockers']}")
    registry["status"] = "sealed_pass"
    for row in registry["datasets"]:
        row["status"] = "sealed"
    write(STUDY / "confirmatory_dataset_registry.json", registry)
    write(STUDY / "confirmatory_dataset_manifest.json", payload)
    write(STUDY / "confirmatory_split_manifest.json", split_payload)
    print(f"PASS: final_datasets_sealed datasets={len(datasets)} labels_accessible=false")


def _identity_hashes(payload: dict[str, object], key: str) -> set[str]:
    direct = payload.get(key)
    if isinstance(direct, list):
        return {str(value) for value in direct}
    relative = payload.get(f"{key}_path")
    if not isinstance(relative, str):
        raise SystemExit(f"FAIL: missing {key} or {key}_path")
    path = (ROOT / relative).resolve()
    if ROOT not in path.parents or not path.is_file():
        raise SystemExit(f"FAIL: invalid identity file for {key}")
    expected = payload.get(f"{key}_file_sha256")
    if expected != sha256(path):
        raise SystemExit(f"FAIL: identity file checksum mismatch for {key}")
    return {line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()}


if __name__ == "__main__":
    main()
