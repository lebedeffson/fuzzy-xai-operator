#!/usr/bin/env python3
"""Freeze final-cycle endpoints and finite formative iteration history."""

from __future__ import annotations

import subprocess

import pandas as pd

from common import BASE, EVIDENCE, ROOT, STUDY, sha256, write


def main() -> None:
    subprocess.run(["git", "merge-base", "--is-ancestor", BASE, "HEAD"], cwd=ROOT, check=True)
    head = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    formative_summary = ROOT / "study/final_confirmatory_closure/formative_real/summary.json"
    primary_comparator = "weighted_linear_score"
    if formative_summary.is_file():
        import json

        measured = json.loads(formative_summary.read_text(encoding="utf-8"))
        primary_comparator = measured.get("best_matched_budget_baseline", {}).get("policy", primary_comparator)
    fixed_risk_budgets = _fixed_risk_budgets()
    protocol = {
        "schema_version": "2.0",
        "identifier": "FXAI-FINAL-ONE-ZIP-PRACTICAL-CLOSURE",
        "base_commit": BASE,
        "implementation_commit": head,
        "phase": "formative_iteration_2_prelock",
        "primary_endpoint": "operationally_invalid_automatic_actions_at_20_percent_review_budget",
        "primary_review_budget": 0.20,
        "secondary_review_budgets": [0.05, 0.10, 0.30],
        "primary_cost_profile": "unsafe_accept_sensitive",
        "primary_comparator_policy": primary_comparator,
        "false_block_ceiling": 0.01,
        "primary_operational_risk_ceiling": 0.05,
        "fixed_risk_operating_budgets_from_development": fixed_risk_budgets,
        "hard_fault_recall_minimum": 0.95,
        "maximum_formative_iterations": 3,
        "current_formative_iteration": 2,
        "confirmatory_test_opened": False,
        "post_lock_changes_forbidden": True,
        "models": {
            "tabular_primary": "sklearn_hist_gradient_boosting",
            "image_primary": "compact_grayscale_cnn",
            "text_primary": "tfidf_sgd_logistic",
            "timeseries_primary": "compact_1d_cnn",
        },
        "calibration_candidates": ["platt", "isotonic", "temperature", "conformal_selective"],
        "p0_schema_width": 10,
        "p1_schema_width": 13,
        "baseline_policies": [
            "always_accept", "always_review", "raw_confidence", "calibrated_confidence",
            "uncertainty", "model_disagreement", "explainer_disagreement", "data_quality",
            "provenance", "simple_or", "weighted_score", "conformal_selective",
            "predictive_risk_only", "full_fuzzyxai",
        ],
        "ambiguity_strata": [
            "low_confidence", "high_confidence_disagreement", "unstable_explanation",
            "incomplete_provenance", "shifted_object", "rare_group", "boundary_object", "route_fault",
        ],
        "success_rules": {
            "H3-P1": {"relative_reduction_minimum": 0.15, "ci_excludes_zero": True, "holm_p_below": 0.05},
            "H3-P2": {"coverage_gain_minimum": 0.05, "ci_excludes_zero": True, "holm_p_below": 0.05},
            "H5-A": {"f1_minimum": 0.95, "false_certification_maximum": 0.01, "source_localization_minimum": 0.90},
            "H7-A": {"canonical_hash_preservation": 1.0},
            "H8": {"action_agreement_minimum": 0.95, "representation_agreement_minimum": 0.90},
            "H9": {"scaling_exponent_maximum": 1.10},
        },
        "holm_families": ["H3", "H5", "H6", "H7", "H8", "H9"],
        "practically_null_intervals": {
            "H3_relative_invalid_action_reduction": [-0.15, 0.15],
            "H3_coverage_gain": [-0.05, 0.05],
            "H7_fidelity_loss": [-0.01, 0.01],
        },
        "immutable_results": {
            "H1": "supported", "H2": "supported", "H3-original": "not_supported", "H4": "supported",
            "H5-S": "supported", "H5-P-original": "not_supported", "H6-general": "not_supported",
        },
    }
    write(STUDY / "protocol.json", protocol)
    iteration = {
        "iterations": [
            {
                "iteration": 1,
                "commit": BASE,
                "result": "completed",
                "config_sha256": _hash_if_present(ROOT / "study/final_practical_closure/practical_protocol.json"),
                "results_sha256": _hash_if_present(ROOT / "release_evidence/final_practical_closure/formative/summary.json"),
                "reason": "initial practical-controller formative cycle",
            },
            {
                "iteration": 2,
                "commit": head,
                "result": "in_progress",
                "config_sha256": sha256(STUDY / "protocol.json"),
                "results_sha256": None,
                "reason": "predeclared confirmatory contracts, H6 estimand correction and run-2 card preparation",
            },
        ],
        "iteration_3_allowed_only_if_predeclared_before_iteration_2_results": True,
    }
    write(STUDY / "formative_iteration_log.json", iteration)
    write(STUDY / "protocol_manifest.json", {"protocol_sha256": sha256(STUDY / "protocol.json"), "test_opened": False})
    write(
        EVIDENCE / "claim_status_prelock.json",
        {
            "phase": "formative_prelock",
            "frozen_claims": protocol["immutable_results"],
            "new_claims": {
                hypothesis: "blocked_pending_sealed_confirmation"
                for hypothesis in ("H3-P1", "H3-P2", "H3-P3", "H3-P4", "H5-A", "H6-A", "H6-B", "H7-A", "H7-B", "H8", "H9")
            },
            "external_claim_gates": {
                "domain_language": "open_external_not_in_scope",
                "comprehension": "open_external_not_in_scope",
                "expert_action": "open_external_not_in_scope",
            },
            "forbidden_release_claims": [
                "understandable_to_users",
                "confirmed_by_experts",
                "improves_domain_safety",
                "matches_specialist_decisions",
            ],
        },
    )
    print("PASS: final_protocol iteration=2/3 confirmatory_opened=false")


def _hash_if_present(path):
    return sha256(path) if path.is_file() else None


def _fixed_risk_budgets() -> dict[str, float | None]:
    path = STUDY / "formative_real/policy_results.parquet"
    if not path.is_file():
        return {}
    frame = pd.read_parquet(path)
    frame = frame[frame["dataset_id"].isna()] if "dataset_id" in frame else frame
    result: dict[str, float | None] = {}
    for policy, rows in frame.groupby("policy"):
        eligible = rows[rows["operational_risk"] <= 0.05]
        result[str(policy)] = float(eligible.sort_values("automatic_coverage").iloc[-1]["review_budget"]) if len(eligible) else None
    return result


if __name__ == "__main__":
    main()
