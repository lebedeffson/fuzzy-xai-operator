from __future__ import annotations

import argparse
from dataclasses import asdict

import numpy as np

from fuzzyxai.practical_controller_v2 import (
    EXPLANATION_FEATURES,
    PREDICTIVE_FEATURES,
    ROUTE_FEATURES,
    SHIFT_FEATURES,
    ActionCostProfile,
    RiskHeadTrainingRow,
    fit_risk_head_oof,
)
from fuzzyxai.replay import stream_events

from .common import ARTIFACTS, git_commit, sha256_file, verify_protocol, write_json


def _feature_rows(count: int = 24_000) -> tuple[dict[str, list[RiskHeadTrainingRow]], dict[str, np.ndarray]]:
    rows = {name: [] for name in ("prediction", "route", "explanation", "shift")}
    targets = {name: [] for name in (*rows, "operational_invalid", "hard_fault")}
    for event in stream_events(count, seed=4201):
        fault_count = len(event.route_faults)
        predictive = {
            "calibrated_confidence": 1.0 - event.confidence,
            "entropy": float(min(1.0, 4.0 * event.confidence * (1.0 - event.confidence))),
            "prediction_margin": 1.0 - abs(2.0 * event.confidence - 1.0),
            "calibration_residual": 0.45 if "stale_calibration" in event.route_faults else 0.03,
            "boundary_distance": 1.0 - abs(2.0 * event.confidence - 1.0),
            "model_disagreement": min(1.0, 0.15 + 0.65 * event.shift_score),
        }
        route = {
            "certificate_exists": float(bool(fault_count)),
            "certificate_coverage": min(1.0, fault_count / 3.0),
            "unsatisfied_contracts": min(1.0, fault_count / 3.0),
            "weighted_contract_severity": min(1.0, (fault_count + int(event.hard_fault)) / 4.0),
            "minimal_cut_size": min(1.0, fault_count / 3.0),
            "path_redundancy": min(1.0, max(0, fault_count - 1) / 2.0),
            "provenance_completeness": float("partial_provenance_loss" in event.route_faults),
            "canonical_integrity": float("canonical_corruption" in event.route_faults),
        }
        explanation = {
            "explainer_disagreement": event.explanation_instability,
            "seed_instability": min(1.0, event.explanation_instability * 0.9),
            "bootstrap_instability": min(1.0, event.explanation_instability * 0.8),
            "perturbation_instability": min(1.0, event.explanation_instability * 1.05),
            "representation_loss": min(1.0, event.explanation_instability * 0.5),
            "source_conflict_count": min(1.0, fault_count / 3.0),
        }
        shift = {
            "shift_score": event.shift_score,
            "reference_population_distance": min(1.0, event.shift_score + 0.2 * ("reference_population_error" in event.route_faults)),
            "artifact_age": float("stale_calibration" in event.route_faults),
            "version_distance": float("model_checkpoint_change" in event.route_faults),
            "rare_group": float(event.sequence_index % 29 == 0),
        }
        target_values = {
            "prediction": event.delayed_label,
            "route": bool(fault_count),
            "explanation": event.explanation_instability >= 0.60,
            "shift": event.shift_score >= 0.55,
        }
        for name, features in (("prediction", predictive), ("route", route), ("explanation", explanation), ("shift", shift)):
            rows[name].append(RiskHeadTrainingRow(event.event_id, f"group-{event.sequence_index // 5}", features, target_values[name]))
            targets[name].append(target_values[name])
        targets["operational_invalid"].append(event.delayed_label or bool(fault_count) or target_values["explanation"] or target_values["shift"])
        targets["hard_fault"].append(event.hard_fault)
    return rows, {name: np.asarray(values, dtype=bool) for name, values in targets.items()}


def _actions(score: np.ndarray, budget: float, hard: np.ndarray) -> np.ndarray:
    result = np.full(len(score), "accept", dtype=object)
    result[hard] = "block"
    candidates = np.flatnonzero(~hard)
    capacity = min(len(candidates), int(budget * len(score)))
    ranked = candidates[np.argsort(-score[candidates], kind="stable")[:capacity]]
    result[ranked] = "review"
    return result


def _metrics(actions: np.ndarray, invalid: np.ndarray, hard: np.ndarray) -> dict[str, float | int]:
    accepted = actions == "accept"
    blocked = actions == "block"
    reviewed = actions == "review"
    return {
        "n_objects": len(actions),
        "coverage": float(np.mean(accepted)),
        "review_rate": float(np.mean(reviewed)),
        "blocked_rate": float(np.mean(blocked)),
        "invalid_automatic_actions": int(np.sum(invalid & accepted)),
        "selective_risk": float(np.sum(invalid & accepted) / max(1, np.sum(accepted))),
        "false_blocks": int(np.sum(blocked & ~hard)),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--iteration", choices=("R1", "R2", "R3"), default="R1")
    parser.add_argument("--objects", type=int, default=24_000)
    args = parser.parse_args()
    verify_protocol()
    rows, targets = _feature_rows(args.objects)
    definitions = (
        ("prediction", "model_error", PREDICTIVE_FEATURES),
        ("route", "route_not_certifiable", ROUTE_FEATURES),
        ("explanation", "explanation_unstable_or_incomplete", EXPLANATION_FEATURES),
        ("shift", "outside_deployment_envelope", SHIFT_FEATURES),
    )
    heads = {}
    scores = {}
    for name, target, features in definitions:
        head, oof = fit_risk_head_oof(rows[name], target_name=target, feature_names=features, calibration_method="platt", seed=4201)
        heads[name] = asdict(head)
        scores[name] = np.asarray(oof)
    predictive = scores["prediction"]
    mixed = 0.55 * predictive + 0.20 * scores["route"] + 0.15 * scores["explanation"] + 0.10 * scores["shift"]
    costs = ActionCostProfile()
    expected_accept = (
        costs.prediction_error * predictive
        + costs.uncertified_route * scores["route"]
        + costs.unstable_explanation * scores["explanation"]
        + costs.deployment_shift * scores["shift"]
    )
    review_loss = np.minimum(costs.short_review + costs.short_review_residual * expected_accept, costs.full_review + costs.full_review_residual * expected_accept)
    benefit = expected_accept - review_loss
    selected_score = {"R1": mixed, "R2": np.maximum.reduce(tuple(scores.values())), "R3": benefit}[args.iteration]
    policy_actions = _actions(selected_score, 0.20, targets["hard_fault"])
    baseline_actions = _actions(predictive, 0.20, np.zeros(len(predictive), dtype=bool))
    output = ARTIFACTS / "formative" / args.iteration.lower()
    output.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output / "oof_scores.npz",
        predictive=scores["prediction"],
        route=scores["route"],
        explanation=scores["explanation"],
        shift=scores["shift"],
        operational_invalid=targets["operational_invalid"],
        hard_fault=targets["hard_fault"],
        policy_actions=policy_actions,
        baseline_actions=baseline_actions,
    )
    summary = {
        "iteration": args.iteration,
        "phase": "formative",
        "objects": args.objects,
        "commit": git_commit(),
        "protocol_sha256": verify_protocol(),
        "features_are_oof": True,
        "test_used_for_selection": False,
        "heads": heads,
        "cost_profile": asdict(costs),
        "controller": _metrics(policy_actions, targets["operational_invalid"], targets["hard_fault"]),
        "predictive_only_baseline": _metrics(baseline_actions, targets["operational_invalid"], targets["hard_fault"]),
        "claim_allowed": False,
        "change_reason": {
            "R1": "separate heads with mixed ranking",
            "R2": "predeclared failure-class correction using maximum head risk",
            "R3": "final formative marginal expected-loss budget optimizer",
        }[args.iteration],
    }
    write_json(output / "summary.json", summary)
    write_json(output / "policy.json", {"policy_version": f"negative-remediation-{args.iteration.lower()}", "heads": heads, "cost_profile": asdict(costs), "selected_without_test": True})
    print(f"PASS remediation-controller-{args.iteration.lower()} objects={args.objects} claim_allowed=false sha256={sha256_file(output / 'summary.json')}")


if __name__ == "__main__":
    main()
