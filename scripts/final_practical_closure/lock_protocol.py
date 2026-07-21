#!/usr/bin/env python3
"""Create the one-way confirmatory lock after every sealed prerequisite passes."""

from __future__ import annotations

from common import FORMATIVE, LOCK, ROOT, STUDY, load_json, sha256, verify_immutable_results, write_json


REQUIRED_MODALITIES = {"tabular": 2, "image": 1, "text": 1, "timeseries": 1}


def main() -> None:
    verify_immutable_results()
    if LOCK.is_file():
        lock = load_json(LOCK)
        if lock.get("status") != "locked" or lock.get("confirmatory_test_opened") is not False:
            raise SystemExit("FAIL: existing confirmatory lock is invalid")
        print(f"PASS: practical_confirmatory_protocol_already_locked datasets={lock['dataset_count']} test_opened=false")
        return
    blockers = _blockers()
    if blockers:
        print("BLOCKED: practical_confirmatory_protocol_lock")
        for blocker in blockers:
            print(f"- {blocker}")
        raise SystemExit(2)
    dataset_path = STUDY / "confirmatory_dataset_manifest.json"
    split_path = STUDY / "confirmatory_split_manifest.json"
    review_path = STUDY / "ai_formative_run2_acceptance.json"
    protocol_path = STUDY / "practical_protocol.json"
    datasets = load_json(dataset_path)["datasets"]
    lock = {
        "schema_version": "1.0",
        "status": "locked",
        "confirmatory_test_opened": False,
        "dataset_count": len(datasets),
        "practical_protocol_sha256": sha256(protocol_path),
        "dataset_manifest_sha256": sha256(dataset_path),
        "split_manifest_sha256": sha256(split_path),
        "ai_formative_run2_sha256": sha256(review_path),
        "formative_evidence_manifest_sha256": sha256(FORMATIVE / "manifest.json"),
        "post_lock_changes_forbidden": True,
    }
    write_json(LOCK, lock)
    print(f"PASS: practical_confirmatory_protocol_locked datasets={len(datasets)} test_opened=false")


def _blockers() -> list[str]:
    required = (
        STUDY / "practical_protocol.json",
        FORMATIVE / "manifest.json",
        STUDY / "confirmatory_dataset_manifest.json",
        STUDY / "confirmatory_split_manifest.json",
        STUDY / "ai_formative_run2_acceptance.json",
    )
    blockers = [f"missing {path.relative_to(ROOT)}" for path in required if not path.is_file()]
    if blockers:
        return blockers
    datasets = load_json(STUDY / "confirmatory_dataset_manifest.json").get("datasets", [])
    splits = load_json(STUDY / "confirmatory_split_manifest.json")
    review = load_json(STUDY / "ai_formative_run2_acceptance.json")
    counts: dict[str, int] = {}
    seen_ids: set[str] = set()
    for item in datasets if isinstance(datasets, list) else []:
        if not isinstance(item, dict):
            blockers.append("dataset manifest contains a non-object entry")
            continue
        dataset_id = str(item.get("dataset_id", ""))
        modality = str(item.get("modality", ""))
        counts[modality] = counts.get(modality, 0) + 1
        if not dataset_id or dataset_id in seen_ids:
            blockers.append(f"invalid or duplicate dataset_id {dataset_id!r}")
        seen_ids.add(dataset_id)
        source = ROOT / str(item.get("sealed_path", ""))
        if not source.is_file() or item.get("sha256") != sha256(source):
            blockers.append(f"invalid sealed dataset {dataset_id or '<unknown>'}")
        if item.get("used_in_formative_tuning") is not False:
            blockers.append(f"dataset {dataset_id} is not independent of formative tuning")
    for modality, minimum in REQUIRED_MODALITIES.items():
        if counts.get(modality, 0) < minimum:
            blockers.append(f"confirmatory {modality} datasets {counts.get(modality, 0)}/{minimum}")
    if splits.get("labels_available_to_tuning_runner") is not False:
        blockers.append("confirmatory labels are not sealed from tuning runner")
    if splits.get("all_features_out_of_fold") is not True:
        blockers.append("controller features are not declared out-of-fold")
    if set(splits.get("dataset_ids", [])) != seen_ids:
        blockers.append("split manifest does not match dataset manifest")
    if review.get("status") != "pass" or review.get("ai_review_is_external_validation") is not False:
        blockers.append("real formative AI-review run 2 has not passed")
    return blockers


if __name__ == "__main__":
    main()

