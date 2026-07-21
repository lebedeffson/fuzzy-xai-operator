#!/usr/bin/env python3
"""Fail-closed verifier for full empirical-validation evidence."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_EVIDENCE = ROOT / "release_evidence/full_empirical_validation"


def read_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise RuntimeError(f"missing required evidence: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise RuntimeError(f"expected JSON object: {path}")
    return payload


def verify_manifest(root: Path) -> None:
    manifest = read_json(root / "manifest_sha256.json")
    for relative, expected in manifest.get("files", {}).items():
        path = root / relative
        if not path.is_file():
            raise RuntimeError(f"manifest file is missing: {relative}")
        actual = hashlib.sha256(path.read_bytes()).hexdigest()
        if actual != expected:
            raise RuntimeError(f"manifest mismatch: {relative}")


def verify(profile: str, root: Path) -> None:
    manifest = read_json(root / "run_manifest.json")
    if manifest.get("profile") != profile:
        raise RuntimeError(f"profile mismatch: expected {profile}, got {manifest.get('profile')}")
    if manifest.get("release_status") != "BLOCKED" or manifest.get("tag_allowed"):
        raise RuntimeError("external gates must block the stable release")
    expected_external = {
        "expert_review": "planned_not_run",
        "comprehension_pilot": "planned_not_run",
        "domain_semantic_review": "pending_external_review",
    }
    if manifest.get("external_gates") != expected_external:
        raise RuntimeError("external gate status was altered without independent evidence")
    gates = {item["experiment_id"]: item for item in manifest.get("experiments", [])}
    if set(gates) != {f"E{index}" for index in range(1, 9)}:
        raise RuntimeError("E1-E8 gate set is incomplete")
    if any(item["status"] == "FAIL" for item in gates.values()):
        raise RuntimeError("an empirical experiment failed")

    multimodal = read_json(root / "empirical_validation/multimodal_results.json")
    datasets = multimodal.get("datasets", [])
    expected_objects = 1000 if profile == "smoke" else 10_000
    if {row["modality"] for row in datasets} != {"tabular", "image", "text", "time_series"}:
        raise RuntimeError("multimodal dataset set is incomplete")
    if any(int(row["n_objects"]) < expected_objects for row in datasets):
        raise RuntimeError("dataset size is below the profile threshold")

    ablation = read_json(root / "rule_ablation/statistical_report.json")
    expected_pairs = 2 if profile == "smoke" else 50
    if int(ablation.get("n_paired_comparisons", 0)) < expected_pairs:
        raise RuntimeError("insufficient paired ablation comparisons")
    if ablation.get("test_partition_used_for_rule_selection"):
        raise RuntimeError("test leakage detected in rule selection")
    if not (root / "rule_ablation/repeated_cv_predictions.csv").is_file():
        raise RuntimeError("object-level ablation predictions are missing")

    baselines = read_json(root / "baselines/statistical_comparison.json")
    if profile == "full" and not baselines.get("all_required_measured"):
        raise RuntimeError("full profile requires measured SHAP, LIME, Anchors, and RuleFit")
    if profile == "full" and not (root / "baselines/object_level_results.parquet").is_file():
        raise RuntimeError("full profile requires baseline object-level Parquet")

    calibration = read_json(root / "calibration/calibration_manifest.json")
    if calibration.get("test_partition_used") or int(calibration.get("trial_count", 0)) != 27:
        raise RuntimeError("calibration is not a complete validation-only grid")
    policies = read_json(root / "policies/policy_comparison.json")
    if len(policies.get("policies", [])) != 21 or not policies.get("costs_predeclared"):
        raise RuntimeError("P1-P7 x three cost scenarios are incomplete")
    sensitivity = read_json(root / "sensitivity/sensitivity_report.json")
    if len(sensitivity.get("parameter_points", [])) != 15 or len(sensitivity.get("input_perturbations", [])) != 4:
        raise RuntimeError("sensitivity contour is incomplete")
    hierarchy = read_json(root / "uncertainty_hierarchy/hierarchy_results.json")
    fml_fraction = float(hierarchy.get("adaptive_fml_fraction", 1.0))
    if fml_fraction > 0.9 and hierarchy.get("practical_hierarchy_claim_allowed"):
        raise RuntimeError("hierarchy benefit claim was not blocked above 90% FML selection")
    critical = read_json(root / "critical_rupture_scalability/critical_rupture_and_scalability.json")
    if len(critical.get("detector_comparison", [])) < 7:
        raise RuntimeError("critical rupture was not compared with required simple signals")
    expected_scaling = 3 if profile == "smoke" else 4
    if len(critical.get("scalability", {}).get("measurements", [])) != expected_scaling:
        raise RuntimeError("scalability size matrix is incomplete")
    population = read_json(root / "object_level/full_population_summary.json")
    if int(population.get("n_objects", 0)) != expected_objects:
        raise RuntimeError("full-population analysis does not cover every object")
    review = read_json(root / "external_review/review_status.json")
    if review.get("status") != "planned_not_run" or review.get("claim_allowed"):
        raise RuntimeError("external expert results were fabricated or mislabelled")
    if int(review.get("sample_size", 0)) != 100:
        raise RuntimeError("expert review sample must contain 100 objects")

    if profile == "full":
        optional = read_json(root / "optional_runtime/multimodal_neural_runtime.json")
        if not all(optional.get("checks", {}).values()):
            raise RuntimeError("CNN/ONNX/text-sequence/time-series-sequence runtime gate failed")
        if not (root / "object_level/all_objects.parquet").is_file():
            raise RuntimeError("full profile requires all-object Parquet")
        if any(gate["status"] != "PASS" for gate in gates.values()):
            raise RuntimeError("full profile requires E1-E8 technical gates to pass")

    verify_manifest(root)
    print(f"PASS: empirical_profile {profile}")
    for experiment_id in sorted(gates):
        print(f"PASS: verified_{experiment_id} status={gates[experiment_id]['status']}")
    print("PASS: manifest_sha256")
    print("BLOCKED: external_expert_review planned_not_run")
    print("BLOCKED: comprehension_pilot planned_not_run")
    print("BLOCKED: stable_release_tag")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--profile", choices=("smoke", "full"), default="smoke")
    parser.add_argument("--root", type=Path, default=DEFAULT_EVIDENCE)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    verify(args.profile, args.root)
