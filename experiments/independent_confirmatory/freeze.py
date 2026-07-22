from __future__ import annotations

from datetime import datetime, timezone

from .common import ARTIFACTS, LOCK, ROOT, git_commit, git_tree_clean, read_json, sha256_file, verify_protocol, write_json
from .modeling import DATASET_IDS


def main() -> None:
    verify_protocol()
    if LOCK.exists():
        raise RuntimeError("confirmatory protocol is already locked")
    if not git_tree_clean():
        raise RuntimeError("freeze requires a clean committed implementation and committed formative evidence")
    model_manifest = ARTIFACTS / "models" / "model_manifest.json"
    formative_path = ARTIFACTS / "formative" / "summary.json"
    data_manifest = ARTIFACTS / "data" / "dataset_manifest.json"
    for path in (model_manifest, formative_path, data_manifest):
        if not path.is_file():
            raise RuntimeError(f"missing prelock artifact: {path.relative_to(ROOT)}")
    formative = read_json(formative_path)
    model = read_json(model_manifest)
    lock = {
        "schema_version": "1.0",
        "protocol_id": "FXAI-Q1-INDEPENDENT-CONFIRMATORY-CLOSURE",
        "locked_at_utc": datetime.now(timezone.utc).isoformat(),
        "implementation_commit": git_commit(),
        "protocol_hashes": verify_protocol(),
        "dataset_ids": list(DATASET_IDS),
        "data_manifest_sha256": sha256_file(data_manifest),
        "model_manifest_sha256": sha256_file(model_manifest),
        "formative_summary_sha256": sha256_file(formative_path),
        "model_artifact_hashes": {item["dataset_id"]: item["model_sha256"] for item in model["datasets"]},
        "observable_split_hashes": {
            item["dataset_id"]: {split: value["artifact_sha256"] for split, value in item["splits"].items()}
            for item in model["datasets"]
        },
        "primary_review_budget": 0.20,
        "best_baseline": formative["best_baseline_selected_without_test"],
        "cost_weights": formative["cost_weights"],
        "policy_score_thresholds": formative["policy_score_thresholds_at_20_percent"],
        "fixed_operational_risk": sum(
            value[formative["best_baseline_selected_without_test"]]["selective_operational_risk"]
            * value[formative["best_baseline_selected_without_test"]]["objects"]
            for value in formative["dataset_results"].values()
        )
        / sum(value[formative["best_baseline_selected_without_test"]]["objects"] for value in formative["dataset_results"].values()),
        "h5_validator_spec_sha256": formative["h5"]["validator_spec_sha256"],
        "h5_held_out_families": formative["h5"]["held_out_family_names_committed_before_confirmatory"],
        "h6_formative_gate": formative["h6"]["gate"],
        "h6_confirmatory_opening_allowed": formative["h6"]["confirmatory_opening_allowed"],
        "h6_blocking_reason": formative["h6"]["blocking_reason"],
        "success_criteria": {
            "H3-R1_relative_reduction_minimum": 0.15,
            "H3-R2_coverage_gain_minimum": 0.05,
            "confidence_interval_excludes_zero": True,
            "holm_alpha": 0.05,
            "hard_block_rate_maximum": 0.05,
            "false_block_rate_maximum": 0.01,
            "H5-A3_unknown_recall_minimum": 0.85,
            "H5-A3_false_certification_maximum": 0.01,
            "H5-A3_unknown_auroc_minimum": 0.90,
            "H5-A3_source_localization_minimum": 0.80,
        },
        "sealed_labels_opened": False,
        "post_lock_tuning_allowed": False,
    }
    write_json(LOCK, lock)
    print(f"PASS independent-freeze implementation={lock['implementation_commit']} best_baseline={lock['best_baseline']} labels_opened=false")


if __name__ == "__main__":
    main()
