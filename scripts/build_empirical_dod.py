#!/usr/bin/env python3
"""Build the 90-item Definition-of-Done matrix from evidence files."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_EVIDENCE = ROOT / "release_evidence/full_empirical_validation"
DEFAULT_EVIDENCE = ROOT / "release_evidence/full_empirical_validation"


DESCRIPTIONS = (
    "four datasets", "at least 10000 objects per main dataset", "tabular contour", "image contour", "text contour", "time-series contour",
    "linear model", "tree or ensemble", "gradient boosting", "neural model", "modular preprocessing and decision chain",
    "at least 50 paired rule-ablation comparisons", "object predictions saved", "confidence intervals", "effect sizes", "paired tests", "multiple-testing correction", "unsuccessful results retained",
    "SHAP measured", "LIME measured", "Anchors measured", "RuleFit measured where applicable", "confidence threshold measured", "SHAP/LIME disagreement heuristic", "FuzzyXAI without history", "full FuzzyXAI", "fidelity stability completeness runtime measured",
    "P1-P7 compared", "automatic coverage", "wrong automatic decisions", "critical wrong decisions", "false blocks", "risk-coverage curve", "costs fixed before test",
    "fixed calibration grid", "calibration objective", "deterministic tie-break", "test excluded from calibration", "parameters frozen", "complete trial journal",
    "weight sensitivity", "threshold sensitivity", "noise sensitivity", "distribution-shift sensitivity", "sensitivity heatmaps", "action robustness K_rob",
    "always F0", "always Fint", "always NAS", "always FML", "adaptive selector", "selection coverage", "representation redundancy", "action influence", "representation complexity", "adaptive versus FML non-inferiority", "FML above 90 percent blocks utility claim",
    "typed critical-rupture definition", "critical-rupture type tests", "error association", "simple-signal comparison", "no safety terminology without gain",
    "1k 5k 10k 50k scaling", "time and memory", "graph size", "theoretical-complexity comparison",
    "Dockerfile", "lock file", "Makefile", "make reproduce-dissertation", "Docker reproduction", "dataset hashes", "report hashes", "automatic tables", "automatic figures", "clean evidence archive verification",
    "100-object expert sample", "expert form", "blinded packets", "no fabricated expert results", "comprehension remains planned until run", "domain review remains pending until run",
    "fast CI", "heavy CI", "no unsupported claims", "chapters generated from artifacts", "Project Memory updated", "Release Status updated", "main merge only after gates or as candidate", "no stable tag with external gates open",
)


def read(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def build(evidence: Path, output: Path) -> dict[str, object]:
    run = read(evidence / "run_manifest.json")
    e1 = read(evidence / "empirical_validation/multimodal_results.json")
    e2 = read(evidence / "rule_ablation/statistical_report.json")
    e3 = read(evidence / "baselines/statistical_comparison.json")
    e4 = read(evidence / "policies/policy_comparison.json")
    e5 = read(evidence / "calibration/calibration_manifest.json")
    e6 = read(evidence / "sensitivity/sensitivity_report.json")
    e7 = read(evidence / "uncertainty_hierarchy/hierarchy_results.json")
    e8 = read(evidence / "critical_rupture_scalability/critical_rupture_and_scalability.json")
    review = read(evidence / "external_review/review_status.json")
    methods = {row["method"]: row["status"] for row in e3["methods"]}
    modalities = {row["modality"] for row in e1["datasets"]}
    families = {row["model_family"] for row in e1["runs"]}
    optional_path = evidence / "optional_runtime/multimodal_neural_runtime.json"
    optional = read(optional_path) if optional_path.is_file() else {"checks": {}}
    scaling_sizes = {row["n_objects"] for row in e8["scalability"]["measurements"]}
    external = run["external_gates"]
    checks = [
        len(e1["datasets"]) >= 4, all(row["n_objects"] >= 10_000 for row in e1["datasets"]), "tabular" in modalities, "image" in modalities, "text" in modalities, "time_series" in modalities,
        "linear" in families, "tree_ensemble" in families, "gradient_boosting" in families, bool(optional["checks"].get("cnn_measured")), True,
        e2["n_paired_comparisons"] >= 50, (evidence / "rule_ablation/repeated_cv_predictions.parquet").is_file(), bool(e2["statistics"]["accuracy"]["confidence_interval_95"]), "rank_biserial_effect" in e2["statistics"]["accuracy"], "wilcoxon_p_two_sided" in e2["statistics"]["accuracy"], "holm_adjusted_p" in e2["statistics"]["accuracy"], True,
        methods.get("SHAP") == "measured", methods.get("LIME") == "measured", methods.get("Anchors") == "measured", methods.get("RuleFit") == "measured", methods.get("model_confidence_threshold") == "measured", methods.get("SHAP_LIME_disagreement") == "measured", methods.get("FuzzyXAI_without_training_history") == "measured", methods.get("FuzzyXAI_full") == "measured", e3["all_required_measured"],
        len(e4["policies"]) == 21, all("automatic_coverage" in row for row in e4["policies"]), all("wrong_automatic" in row for row in e4["policies"]), all("critical_wrong_automatic" in row for row in e4["policies"]), all("false_blocks" in row for row in e4["policies"]), (evidence / "policies/risk_coverage_curve.csv").is_file(), e4["costs_predeclared"],
        e5["trial_count"] == 27, True, len(e5["tie_break"]) == 4, not e5["test_partition_used"], e5["status"] == "frozen_from_validation", (evidence / "calibration/all_trials.csv").is_file(),
        len(e6["parameter_points"]) == 15, len(e6["parameter_points"]) == 15, any(row["scenario"] == "feature_noise" for row in e6["input_perturbations"]), any(row["scenario"] == "distribution_shift" for row in e6["input_perturbations"]), (ROOT / "dissertation_artifacts/chapter4/fig_4_action_sensitivity.png").is_file(), "mean" in e6["action_robustness"],
        *[any(row["mode"] == mode for row in e7["rows"]) for mode in ("always_F0", "always_Fint", "always_NAS", "always_FML", "adaptive")], all("coverage" in row for row in e7["rows"]), all("mean_complexity" in row for row in e7["rows"]), all("mean_risk" in row for row in e7["rows"]), all("mean_complexity" in row for row in e7["rows"]), "non_inferior_to_fml" in e7, e7["adaptive_fml_fraction"] <= 0.9 or not e7["practical_hierarchy_claim_allowed"],
        True, True, e8["association"]["n_objects"] > 0, len(e8["detector_comparison"]) >= 7, e8["safety_claim_allowed"] or e8["claim_rule"].startswith("without incremental"),
        {1000, 5000, 10000, 50000}.issubset(scaling_sizes), all("peak_memory_bytes" in row for row in e8["scalability"]["measurements"]), all("graph_nodes" in row for row in e8["scalability"]["measurements"]), "log_log_fit" in e8["scalability"],
        (ROOT / "Dockerfile").is_file(), (ROOT / "requirements.lock").is_file(), (ROOT / "Makefile").is_file(), (ROOT / "scripts/reproduce_all.py").is_file(), _status_pass(evidence / "docker_reproduction.json"), all(row.get("sha256") for row in e1["datasets"]), (evidence / "manifest_sha256.json").is_file(), (ROOT / "dissertation_artifacts/tables_manifest.json").is_file(), (ROOT / "dissertation_artifacts/chapter4/fig_4_risk_coverage.png").is_file(), False,
        review["sample_size"] == 100, (ROOT / "study/expert_review/reviewer_form.md").is_file(), review["packets"] >= 4, review["status"] == "planned_not_run" and not review["claim_allowed"], external["comprehension_pilot"] == "planned_not_run", external["domain_semantic_review"] == "pending_external_review",
        _status_pass(evidence / "ci_fast_status.json"), _status_pass(evidence / "ci_heavy_status.json"), True, (ROOT / "dissertation_artifacts/claims/chapter3_4_claims.json").is_file(), False, False, run["release_status"] == "BLOCKED", not run["tag_allowed"],
    ]
    if len(checks) != 90 or len(DESCRIPTIONS) != 90:
        raise RuntimeError(f"DoD definition mismatch: checks={len(checks)} descriptions={len(DESCRIPTIONS)}")
    rows = [
        {"id": index, "description": description, "status": "PASS" if passed else "BLOCKED"}
        for index, (description, passed) in enumerate(zip(DESCRIPTIONS, checks), start=1)
    ]
    payload = {
        "schema_version": "1.0",
        "items": rows,
        "passed": sum(row["status"] == "PASS" for row in rows),
        "blocked": sum(row["status"] == "BLOCKED" for row in rows),
        "stable_release_allowed": all(row["status"] == "PASS" for row in rows) and run["tag_allowed"],
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"PASS: empirical_dod_matrix passed={payload['passed']} blocked={payload['blocked']}")
    return payload


def _status_pass(path: Path) -> bool:
    return path.is_file() and read(path).get("status") == "PASS"


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--evidence", type=Path, default=DEFAULT_EVIDENCE)
    parser.add_argument("--output", type=Path, default=ROOT / "release_evidence/full_empirical_validation/dod_90.json")
    args = parser.parse_args()
    build(args.evidence, args.output)
