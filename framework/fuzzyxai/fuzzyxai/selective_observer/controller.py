"""Calibrated selective controller trained only from out-of-fold development evidence."""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict
from typing import Sequence

import numpy as np

from .contracts import (
    ConfirmatoryExample,
    ConfirmatoryProtocolLock,
    DevelopmentExample,
    FEATURE_NAMES,
    PolicyMetrics,
    SelectiveAction,
    SelectiveControllerSpec,
    SelectiveRiskFeatures,
)


def fit_selective_controller(
    examples: Sequence[DevelopmentExample],
    *,
    folds: int = 5,
    seed: int = 4201,
    baseline_confidence_threshold: float = 0.75,
) -> tuple[SelectiveControllerSpec, dict[str, object]]:
    """Fit a second-stage risk model and choose thresholds from OOF predictions."""
    if len(examples) < max(20, folds * 4):
        raise ValueError("at least 20 development examples are required")
    matrix = np.asarray([example.features.to_vector() for example in examples], dtype=float)
    labels = np.asarray([example.unsafe_automatic_action for example in examples], dtype=int)
    if len(np.unique(labels)) != 2:
        raise ValueError("development examples must contain both outcome classes")
    means = matrix.mean(axis=0)
    scales = matrix.std(axis=0)
    scales[scales <= 1e-12] = 1.0
    normalized = (matrix - means) / scales
    groups = np.asarray([example.group_id for example in examples])
    oof_risk = _oof_probabilities(normalized, labels, groups, folds=folds, seed=seed)
    coefficients, intercept = _fit_logistic(normalized, labels, seed=seed)

    baseline_accept = matrix[:, 0] <= 1.0 - baseline_confidence_threshold
    baseline = _binary_metrics(labels, baseline_accept)
    accept_threshold = _select_accept_threshold(oof_risk, labels, target_coverage=float(baseline["coverage"]))
    upper = oof_risk[oof_risk > accept_threshold]
    short_threshold = max(accept_threshold, float(np.quantile(upper, 0.50)) if len(upper) else accept_threshold)
    full_threshold = max(short_threshold, float(np.quantile(upper, 0.85)) if len(upper) else short_threshold)
    development_hash = _development_hash(examples)
    spec = SelectiveControllerSpec(
        schema_version="1.0",
        feature_names=FEATURE_NAMES,
        coefficients=tuple(float(value) for value in coefficients),
        intercept=float(intercept),
        feature_means=tuple(float(value) for value in means),
        feature_scales=tuple(float(value) for value in scales),
        accept_max_risk=accept_threshold,
        short_review_max_risk=short_threshold,
        full_review_max_risk=full_threshold,
        block_rupture_severity=0.95,
        development_hash=development_hash,
        selected_without_test=True,
    )
    controller_accept = oof_risk <= accept_threshold
    controller = _binary_metrics(labels, controller_accept)
    relative_reduction = _relative_reduction(float(baseline["selective_risk"]), float(controller["selective_risk"]))
    return spec, {
        "phase": "formative",
        "selection_predictions": "controller out-of-fold",
        "source_features": "base-model out-of-fold",
        "baseline": baseline,
        "controller": controller,
        "coverage_difference": float(controller["coverage"]) - float(baseline["coverage"]),
        "relative_wrong_automatic_reduction": relative_reduction,
        "formative_target_met": abs(float(controller["coverage"]) - float(baseline["coverage"])) <= 0.02 and relative_reduction >= 0.15,
        "confirmatory_claim_allowed": False,
    }


def predict_risk(spec: SelectiveControllerSpec, features: SelectiveRiskFeatures) -> float:
    values = np.asarray(features.to_vector(), dtype=float)
    means = np.asarray(spec.feature_means, dtype=float)
    scales = np.asarray(spec.feature_scales, dtype=float)
    score = float(np.dot((values - means) / scales, np.asarray(spec.coefficients)) + spec.intercept)
    return float(1.0 / (1.0 + np.exp(-np.clip(score, -40.0, 40.0))))


def decide(spec: SelectiveControllerSpec, features: SelectiveRiskFeatures) -> SelectiveAction:
    if features.rupture_severity >= spec.block_rupture_severity or features.provenance_incompleteness >= 0.99:
        return SelectiveAction.BLOCK
    risk = predict_risk(spec, features)
    if risk <= spec.accept_max_risk:
        return SelectiveAction.ACCEPT
    if risk <= spec.short_review_max_risk:
        return SelectiveAction.SHORT_REVIEW
    if risk <= spec.full_review_max_risk:
        return SelectiveAction.FULL_REVIEW
    return SelectiveAction.BLOCK


def evaluate_confirmatory_policy(
    spec: SelectiveControllerSpec,
    examples: Sequence[ConfirmatoryExample],
    protocol_lock: ConfirmatoryProtocolLock,
) -> dict[str, object]:
    if not examples:
        raise ValueError("confirmatory evaluation requires frozen test examples")
    actions = [decide(spec, example.features) for example in examples]
    metrics = policy_metrics([example.unsafe_automatic_action for example in examples], actions)
    return {
        "phase": "confirmatory",
        "controller_development_hash": spec.development_hash,
        "protocol_sha256": protocol_lock.protocol_sha256,
        "n_objects": len(examples),
        "metrics": asdict(metrics),
        "thresholds_frozen_before_test": True,
    }


def policy_metrics(outcomes: Sequence[bool], actions: Sequence[SelectiveAction]) -> PolicyMetrics:
    if len(outcomes) != len(actions) or not outcomes:
        raise ValueError("outcomes and actions must be non-empty and aligned")
    wrong = np.asarray(outcomes, dtype=bool)
    accepted = np.asarray([action is SelectiveAction.ACCEPT for action in actions])
    reviewed = ~accepted
    accepted_count = int(accepted.sum())
    wrong_automatic = int(np.sum(wrong & accepted))
    return PolicyMetrics(
        n_objects=len(actions),
        coverage=accepted_count / len(actions),
        selective_risk=wrong_automatic / max(1, accepted_count),
        wrong_automatic=wrong_automatic,
        short_review=sum(action is SelectiveAction.SHORT_REVIEW for action in actions),
        full_review=sum(action is SelectiveAction.FULL_REVIEW for action in actions),
        blocked=sum(action is SelectiveAction.BLOCK for action in actions),
        manual_review_fraction=float(reviewed.mean()),
    )


def risk_coverage_curve(risk_scores: Sequence[float], outcomes: Sequence[bool]) -> list[dict[str, float | int]]:
    scores = np.asarray(risk_scores, dtype=float)
    labels = np.asarray(outcomes, dtype=bool)
    if len(scores) != len(labels) or not len(scores):
        raise ValueError("risk scores and outcomes must be non-empty and aligned")
    thresholds = np.unique(np.concatenate(([0.0], scores, [1.0])))
    result = []
    for threshold in thresholds:
        accepted = scores <= threshold
        accepted_count = int(accepted.sum())
        result.append(
            {
                "threshold": float(threshold),
                "coverage": accepted_count / len(scores),
                "selective_risk": int(np.sum(labels & accepted)) / max(1, accepted_count),
                "wrong_automatic": int(np.sum(labels & accepted)),
                "review_fraction": float((~accepted).mean()),
            }
        )
    return result


def matched_coverage_outcome(
    controller: PolicyMetrics,
    baseline: PolicyMetrics,
    *,
    coverage_tolerance: float = 0.02,
    minimum_relative_risk_reduction: float = 0.15,
) -> dict[str, object]:
    coverage_matched = abs(controller.coverage - baseline.coverage) <= coverage_tolerance
    reduction = _relative_reduction(baseline.selective_risk, controller.selective_risk)
    return {
        "coverage_matched": coverage_matched,
        "relative_risk_reduction": reduction,
        "criterion_met": coverage_matched and reduction >= minimum_relative_risk_reduction,
        "claim_allowed": False,
        "reason": "formative comparison cannot support a confirmatory claim",
    }


def _fit_logistic(matrix: np.ndarray, labels: np.ndarray, *, seed: int) -> tuple[np.ndarray, float]:
    from sklearn.linear_model import LogisticRegression

    model = LogisticRegression(max_iter=1000, class_weight="balanced", random_state=seed)
    model.fit(matrix, labels)
    return np.asarray(model.coef_[0], dtype=float), float(model.intercept_[0])


def _oof_probabilities(
    matrix: np.ndarray,
    labels: np.ndarray,
    groups: np.ndarray,
    *,
    folds: int,
    seed: int,
) -> np.ndarray:
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import StratifiedGroupKFold

    counts = np.bincount(labels)
    effective_folds = min(folds, int(counts.min()), len(np.unique(groups)))
    if effective_folds < 2:
        raise ValueError("each outcome class requires at least two development examples")
    splitter = StratifiedGroupKFold(n_splits=effective_folds, shuffle=True, random_state=seed)
    result: np.ndarray = np.zeros(len(labels), dtype=float)
    for train, validation in splitter.split(matrix, labels, groups):
        model = LogisticRegression(max_iter=1000, class_weight="balanced", random_state=seed)
        model.fit(matrix[train], labels[train])
        result[validation] = model.predict_proba(matrix[validation])[:, 1]
    return result


def _select_accept_threshold(scores: np.ndarray, labels: np.ndarray, *, target_coverage: float) -> float:
    candidates = risk_coverage_curve(scores.tolist(), labels.astype(bool).tolist())
    feasible = [row for row in candidates if abs(float(row["coverage"]) - target_coverage) <= 0.02]
    if not feasible:
        feasible = sorted(candidates, key=lambda row: abs(float(row["coverage"]) - target_coverage))[:1]
    best = min(feasible, key=lambda row: (float(row["selective_risk"]), -float(row["coverage"])))
    return float(best["threshold"])


def _binary_metrics(labels: np.ndarray, accepted: np.ndarray) -> dict[str, float | int]:
    accepted_count = int(accepted.sum())
    wrong = int(np.sum((labels == 1) & accepted))
    return {
        "coverage": accepted_count / len(labels),
        "selective_risk": wrong / max(1, accepted_count),
        "wrong_automatic": wrong,
        "review_fraction": float((~accepted).mean()),
    }


def _relative_reduction(baseline: float, candidate: float) -> float:
    return (baseline - candidate) / baseline if baseline > 0 else 0.0


def _development_hash(examples: Sequence[DevelopmentExample]) -> str:
    payload = [
        {
            "object_id": item.object_id,
            "features": item.features.to_vector(),
            "outcome": item.unsafe_automatic_action,
            "partition": item.partition.value,
            "group_id": item.group_id,
        }
        for item in examples
    ]
    return hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
