from __future__ import annotations

from dataclasses import asdict

import numpy as np

from fuzzyxai.open_set_validator import OpenSetTrainingRow, StructuralObservation, fit_open_set_validator
from fuzzyxai.rule_effects_v2 import (
    audit_conditional_sampler,
    binary_effect_power,
    cluster_equivalent_rules,
    cross_fitted_doubly_robust_effect,
    evaluate_h6_formative_gate,
)

from .common import ARTIFACTS, PRIVATE, git_commit, sha256_file, verify_protocol, write_json
from .modeling import DATASET_IDS


POLICY_NAMES = (
    "raw_confidence",
    "entropy_threshold",
    "predictive_only",
    "simple_or",
    "weighted_score",
    "route_only",
    "deterministic_random",
    "full_hierarchical_fuzzyxai",
)


def _load_rows(dataset_id: str, split: str) -> dict[str, np.ndarray]:
    path = PRIVATE / f"{dataset_id}-{split}.npz"
    if not path.is_file():
        raise RuntimeError(f"missing model output: {path}")
    with np.load(path, allow_pickle=False) as data:
        return {name: data[name] for name in data.files}


def policy_scores(rows: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    predictive = rows["predictive_risk"].astype(float)
    entropy = rows["entropy"].astype(float)
    route = rows["route_risk"].astype(float)
    explanation = rows["explanation_risk"].astype(float)
    shift = rows["shift_risk"].astype(float)
    random_score = np.asarray(
        [int.from_bytes(__import__("hashlib").sha256(f"policy-random:{value}".encode()).digest()[:8], "big") / 2**64 for value in rows["object_ids"]]
    )
    return {
        "raw_confidence": predictive,
        "entropy_threshold": entropy,
        "predictive_only": 0.65 * predictive + 0.20 * entropy + 0.15 * rows["margin_risk"].astype(float),
        "simple_or": np.maximum.reduce((route, (explanation >= 0.80).astype(float), (shift >= 0.90).astype(float), predictive)),
        "weighted_score": 0.60 * predictive + 0.15 * route + 0.15 * explanation + 0.10 * shift,
        "route_only": route + 1e-3 * predictive,
        "deterministic_random": random_score,
        "full_hierarchical_fuzzyxai": 4.0 * predictive + 3.0 * route + 1.0 * explanation + 1.5 * shift,
    }


def assign_actions(rows: dict[str, np.ndarray], policy_name: str, *, budget: float = 0.20) -> np.ndarray:
    scores = policy_scores(rows)[policy_name]
    count = len(scores)
    actions = np.full(count, "accept", dtype="U18")
    if policy_name == "full_hierarchical_fuzzyxai":
        irreparable = rows["irreparable_fault"].astype(bool)
        repairable = rows["repairable_fault"].astype(bool)
        actions[irreparable] = "block"
        actions[repairable] = "repair_then_retry"
        review_capacity = max(0, int(np.floor(budget * count)) - int(np.sum(repairable)))
        candidates = np.flatnonzero(~irreparable & ~repairable)
    else:
        review_capacity = int(np.floor(budget * count))
        candidates = np.arange(count)
    selected = candidates[np.argsort(-scores[candidates], kind="stable")[:review_capacity]]
    high_explanation = rows["explanation_risk"][selected] >= 0.70
    actions[selected[high_explanation]] = "full_review"
    actions[selected[~high_explanation]] = "short_review"
    return actions


def metric_row(rows: dict[str, np.ndarray], actions: np.ndarray) -> dict[str, float | int]:
    if "labels" not in rows:
        raise RuntimeError("development labels are required for formative policy selection")
    model_error = rows["predictions"].astype(str) != rows["labels"].astype(str)
    operational_invalid = model_error | rows["route_risk"].astype(bool) | (rows["explanation_risk"] >= 0.95) | (rows["shift_risk"] >= 0.97)
    accepted = actions == "accept"
    reviewed = np.isin(actions, ("short_review", "full_review", "repair_then_retry"))
    blocked = actions == "block"
    return {
        "objects": len(actions),
        "coverage": float(np.mean(accepted)),
        "review_equivalent_rate": float(np.mean(reviewed)),
        "hard_block_rate": float(np.mean(blocked)),
        "false_block_rate": float(np.mean(blocked & ~rows["irreparable_fault"].astype(bool))),
        "invalid_automatic_actions": int(np.sum(operational_invalid & accepted)),
        "model_errors_accepted": int(np.sum(model_error & accepted)),
        "selective_operational_risk": float(np.mean(operational_invalid[accepted])) if np.any(accepted) else 0.0,
    }


FEATURE_NAMES = (
    "artifact_version_distance",
    "canonical_failure",
    "cross_model_mix",
    "feature_space_mismatch",
    "provenance_loss",
    "reduction_link_break",
    "reference_distance",
    "type_mismatch",
    "unexpected_depth",
    "unseen_edge_ratio",
)
REGIONS = {
    "artifact_version_distance": "model_registry",
    "canonical_failure": "canonical_store",
    "cross_model_mix": "artifact_link",
    "feature_space_mismatch": "feature_schema",
    "provenance_loss": "provenance_graph",
    "reduction_link_break": "representation_reducer",
    "reference_distance": "reference_population",
    "type_mismatch": "typed_contract",
    "unexpected_depth": "route_graph",
    "unseen_edge_ratio": "route_graph",
}
KNOWN_PATTERNS = {
    "version_mismatch": {"artifact_version_distance": 3.0, "type_mismatch": 0.8},
    "missing_provenance": {"provenance_loss": 3.0, "reference_distance": 0.5},
    "canonical_corruption": {"canonical_failure": 3.0, "type_mismatch": 0.4},
    "stale_reference": {"reference_distance": 3.0, "artifact_version_distance": 0.4},
}
UNKNOWN_PATTERNS = {
    "heldout_graph_reorder": {"unseen_edge_ratio": 4.0, "unexpected_depth": 2.0},
    "heldout_cross_model_mix": {"cross_model_mix": 4.0, "type_mismatch": 2.0},
    "heldout_feature_space": {"feature_space_mismatch": 4.0, "reduction_link_break": 1.5},
    "heldout_reduction_link": {"reduction_link_break": 4.0, "unexpected_depth": 1.5},
}


def structural_observation(index: int, family: str, *, partition: str, seed: int) -> StructuralObservation:
    rng = np.random.default_rng(seed + index * 17)
    values = {name: float(rng.normal(0.0, 0.08)) for name in FEATURE_NAMES}
    pattern = {**KNOWN_PATTERNS, **UNKNOWN_PATTERNS}.get(family, {})
    for name, magnitude in pattern.items():
        values[name] += magnitude + float(rng.normal(0.0, 0.12))
    return StructuralObservation(
        f"{partition}-{family}-{index:05d}",
        values,
        REGIONS,
        source_is_oof=partition == "development",
        partition=partition,
    )


def run_h5_formative() -> dict[str, object]:
    rows = []
    for family in ("valid_route", *KNOWN_PATTERNS):
        for index in range(240):
            rows.append(OpenSetTrainingRow(structural_observation(index, family, partition="development", seed=8101), family))
    spec = fit_open_set_validator(rows, known_quantile=0.99, valid_energy_quantile=0.99)
    output = ARTIFACTS / "formative" / "h5_open_set"
    write_json(output / "validator_spec.json", asdict(spec))
    return {
        "known_development_families": sorted(KNOWN_PATTERNS),
        "held_out_family_names_committed_before_confirmatory": sorted(UNKNOWN_PATTERNS),
        "development_rows": len(rows),
        "validator_spec_sha256": sha256_file(output / "validator_spec.json"),
        "test_used": False,
    }


def run_h6_formative() -> dict[str, object]:
    rng = np.random.default_rng(9101)
    detections = []
    null_detections = []
    sign_matches = []
    eligible = []
    effects = (0.05, 0.08, 0.12)
    supports = (0.10, 0.20)
    for scenario, (effect, support) in enumerate((pair for effect in effects for pair in ((effect, supports[0]), (effect, supports[1])))):
        count = 3000
        covariates = rng.normal(size=(count, 5))
        logits = np.log(support / (1.0 - support)) + 0.35 * covariates[:, 0] - 0.25 * covariates[:, 1]
        treatment = rng.random(count) < (1.0 / (1.0 + np.exp(-logits)))
        outcome = 0.4 * covariates[:, 0] - 0.2 * covariates[:, 2] + effect * treatment + rng.normal(0.0, 0.35, count)
        estimate = cross_fitted_doubly_robust_effect(covariates, treatment, outcome, folds=3, seed=9101 + scenario)
        power = binary_effect_power(effect=effect, support=float(np.mean(treatment)), total_objects=count)
        eligible.append(power.eligible)
        detections.append(estimate.confidence_interval_95[0] > 0.0)
        sign_matches.append(estimate.effect > 0.0)
    for scenario in range(6):
        count = 3000
        covariates = rng.normal(size=(count, 5))
        treatment = rng.random(count) < 0.20
        outcome = 0.4 * covariates[:, 0] - 0.2 * covariates[:, 2] + rng.normal(0.0, 0.35, count)
        estimate = cross_fitted_doubly_robust_effect(covariates, treatment, outcome, folds=3, seed=9201 + scenario)
        null_detections.append(not (estimate.confidence_interval_95[0] <= 0.0 <= estimate.confidence_interval_95[1]))
    reference = rng.normal(size=(800, 4))
    generated = reference[rng.integers(0, len(reference), size=800)] + rng.normal(0.0, 0.03, (800, 4))
    sampler = audit_conditional_sampler(reference, generated)
    activations = {
        "rule-a": reference[:, 0] > 0.2,
        "rule-a-proxy": reference[:, 0] + 0.05 * reference[:, 1] > 0.2,
        "rule-b": reference[:, 2] < -0.4,
    }
    clusters = cluster_equivalent_rules(activations, jaccard_threshold=0.80)
    gate = evaluate_h6_formative_gate(
        detection_rate=float(np.mean(detections)),
        sign_accuracy=float(np.mean(sign_matches)),
        false_discovery_rate=float(np.mean(null_detections)),
        power_eligible_fraction=float(np.mean(eligible)),
    )
    return {
        "gate": asdict(gate),
        "true_effect_scenarios": len(detections),
        "null_scenarios": len(null_detections),
        "conditional_sampler_audit": asdict(sampler),
        "equivalence_clusters": [asdict(item) for item in clusters],
        "confirmatory_independent_tabular_datasets": 1,
        "confirmatory_opening_allowed": bool(gate.passed and sampler.passed and False),
        "blocking_reason": "requires at least two independent tabular datasets even if the synthetic formative gate passes",
    }


def main() -> None:
    verify_protocol()
    dataset_results = {}
    aggregate = {name: {"invalid": 0, "objects": 0} for name in POLICY_NAMES}
    score_parts = {name: [] for name in POLICY_NAMES}
    for dataset_id in DATASET_IDS:
        rows = _load_rows(dataset_id, "formative_development")
        policies = {}
        for name in POLICY_NAMES:
            metrics = metric_row(rows, assign_actions(rows, name))
            policies[name] = metrics
            aggregate[name]["invalid"] += int(metrics["invalid_automatic_actions"])
            aggregate[name]["objects"] += int(metrics["objects"])
            score_parts[name].append(policy_scores(rows)[name])
        dataset_results[dataset_id] = policies
    baselines = tuple(name for name in POLICY_NAMES if name != "full_hierarchical_fuzzyxai")
    best_baseline = min(baselines, key=lambda name: (aggregate[name]["invalid"], name))
    h5 = run_h5_formative()
    h6 = run_h6_formative()
    result = {
        "schema_version": "1.0",
        "phase": "formative_development_only",
        "implementation_commit": git_commit(),
        "review_budget": 0.20,
        "policies": list(POLICY_NAMES),
        "dataset_results": dataset_results,
        "aggregate": aggregate,
        "best_baseline_selected_without_test": best_baseline,
        "cost_weights": {"prediction": 4.0, "route": 3.0, "explanation": 1.0, "shift": 1.5},
        "policy_score_thresholds_at_20_percent": {
            name: float(np.quantile(np.concatenate(values), 0.80)) for name, values in score_parts.items()
        },
        "h5": h5,
        "h6": h6,
        "sealed_calibration_labels_loaded": False,
        "sealed_confirmatory_labels_loaded": False,
    }
    write_json(ARTIFACTS / "formative" / "summary.json", result)
    print(f"PASS independent-formative best_baseline={best_baseline} h6_open={str(h6['confirmatory_opening_allowed']).lower()} test_labels=false")


if __name__ == "__main__":
    main()
