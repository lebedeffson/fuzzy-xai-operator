"""Preregisterable H3 baselines and operating-point comparisons."""

from __future__ import annotations

from dataclasses import asdict
from typing import Mapping, Sequence

import numpy as np

from .contracts import SelectiveAction, SelectiveRiskFeatures
from .controller import policy_metrics


def confidence_threshold_policy(
    features: Sequence[SelectiveRiskFeatures],
    *,
    confidence_threshold: float,
) -> list[SelectiveAction]:
    uncertainty_limit = 1.0 - confidence_threshold
    return [SelectiveAction.ACCEPT if item.model_uncertainty <= uncertainty_limit else SelectiveAction.SHORT_REVIEW for item in features]


def uncertainty_policy(
    features: Sequence[SelectiveRiskFeatures],
    *,
    uncertainty_threshold: float,
) -> list[SelectiveAction]:
    return [
        SelectiveAction.ACCEPT
        if max(item.model_uncertainty, item.calibration_residual, item.attribution_instability) <= uncertainty_threshold
        else SelectiveAction.SHORT_REVIEW
        for item in features
    ]


def explainer_disagreement_policy(
    features: Sequence[SelectiveRiskFeatures],
    *,
    confidence_threshold: float,
    disagreement_threshold: float,
) -> list[SelectiveAction]:
    uncertainty_limit = 1.0 - confidence_threshold
    return [
        SelectiveAction.ACCEPT
        if item.model_uncertainty <= uncertainty_limit and item.explainer_disagreement <= disagreement_threshold
        else SelectiveAction.FULL_REVIEW
        for item in features
    ]


def selective_risk_control_policy(
    nonconformity_scores: Sequence[float],
    *,
    frozen_threshold: float,
) -> list[SelectiveAction]:
    """Represent a validation-calibrated conformal/selective-risk baseline."""
    if not 0.0 <= frozen_threshold <= 1.0:
        raise ValueError("frozen nonconformity threshold must be in [0, 1]")
    return [SelectiveAction.ACCEPT if 0.0 <= float(score) <= frozen_threshold else SelectiveAction.FULL_REVIEW for score in nonconformity_scores]


def compare_preregistered_baselines(
    outcomes: Sequence[bool],
    controller_actions: Sequence[SelectiveAction],
    baseline_actions: Mapping[str, Sequence[SelectiveAction]],
    *,
    coverage_tolerance: float = 0.02,
    risk_tolerance: float = 0.005,
    minimum_relative_risk_reduction: float = 0.15,
    minimum_coverage_gain: float = 0.05,
) -> dict[str, object]:
    if not baseline_actions:
        raise ValueError("at least one preregistered baseline is required")
    controller = policy_metrics(outcomes, controller_actions)
    baselines = {name: policy_metrics(outcomes, actions) for name, actions in baseline_actions.items()}

    coverage_comparators = [(name, metrics) for name, metrics in baselines.items() if abs(metrics.coverage - controller.coverage) <= coverage_tolerance]
    matched_coverage = None
    if coverage_comparators:
        name, strongest = min(coverage_comparators, key=lambda item: item[1].selective_risk)
        relative_reduction = _relative_reduction(strongest.selective_risk, controller.selective_risk)
        matched_coverage = {
            "baseline": name,
            "coverage_difference": controller.coverage - strongest.coverage,
            "relative_risk_reduction": relative_reduction,
            "criterion_met": relative_reduction >= minimum_relative_risk_reduction,
        }

    risk_comparators = [(name, metrics) for name, metrics in baselines.items() if abs(metrics.selective_risk - controller.selective_risk) <= risk_tolerance]
    matched_risk = None
    if risk_comparators:
        name, strongest = max(risk_comparators, key=lambda item: item[1].coverage)
        coverage_gain = controller.coverage - strongest.coverage
        matched_risk = {
            "baseline": name,
            "risk_difference": controller.selective_risk - strongest.selective_risk,
            "coverage_gain": coverage_gain,
            "criterion_met": coverage_gain >= minimum_coverage_gain,
        }

    criterion_met = bool((matched_coverage and matched_coverage["criterion_met"]) or (matched_risk and matched_risk["criterion_met"]))
    return {
        "controller": asdict(controller),
        "baselines": {name: asdict(metrics) for name, metrics in baselines.items()},
        "matched_coverage": matched_coverage,
        "matched_risk": matched_risk,
        "criterion_met": criterion_met,
        "claim_allowed": False,
        "reason": "operating points are formative until the preregistered confirmatory test is opened",
    }


def cost_review_points(
    outcomes: Sequence[bool],
    policies: Mapping[str, Sequence[SelectiveAction]],
    *,
    wrong_automatic_cost: float,
    short_review_cost: float,
    full_review_cost: float,
    block_cost: float,
) -> list[dict[str, float | str]]:
    if min(wrong_automatic_cost, short_review_cost, full_review_cost, block_cost) < 0:
        raise ValueError("costs cannot be negative")
    result: list[dict[str, float | str]] = []
    for name, actions in policies.items():
        metrics = policy_metrics(outcomes, actions)
        total = (
            wrong_automatic_cost * metrics.wrong_automatic
            + short_review_cost * metrics.short_review
            + full_review_cost * metrics.full_review
            + block_cost * metrics.blocked
        )
        result.append(
            {
                "policy": name,
                "manual_review_fraction": metrics.manual_review_fraction,
                "mean_cost": float(total / metrics.n_objects),
            }
        )
    return result


def _relative_reduction(baseline: float, candidate: float) -> float:
    if np.isclose(baseline, 0.0):
        return 0.0
    return (baseline - candidate) / baseline
