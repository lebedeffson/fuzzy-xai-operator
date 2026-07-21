"""Metrics used by empirical protocols without embedding scientific claims."""

from __future__ import annotations

import math
from typing import Sequence

import numpy as np


def binary_classification_metrics(
    labels: Sequence[int],
    probabilities: Sequence[float],
    *,
    subgroup_mask: Sequence[bool] | None = None,
    bins: int = 10,
) -> dict[str, float | int]:
    truth = np.asarray(labels, dtype=int)
    scores = np.asarray(probabilities, dtype=float)
    predictions = (scores >= 0.5).astype(int)
    if len(truth) != len(scores) or not len(truth):
        raise ValueError("labels and probabilities must be non-empty and aligned")
    true_positive = int(np.sum((truth == 1) & (predictions == 1)))
    true_negative = int(np.sum((truth == 0) & (predictions == 0)))
    false_positive = int(np.sum((truth == 0) & (predictions == 1)))
    false_negative = int(np.sum((truth == 1) & (predictions == 0)))
    precision = _safe_ratio(true_positive, true_positive + false_positive)
    recall = _safe_ratio(true_positive, true_positive + false_negative)
    specificity = _safe_ratio(true_negative, true_negative + false_positive)
    result: dict[str, float | int] = {
        "n_objects": len(truth),
        "accuracy": float(np.mean(truth == predictions)),
        "balanced_accuracy": (recall + specificity) / 2.0,
        "precision": precision,
        "recall": recall,
        "f1": _safe_ratio(2.0 * precision * recall, precision + recall),
        "auroc": binary_auroc(truth, scores),
        "expected_calibration_error": expected_calibration_error(truth, scores, bins=bins),
        "critical_errors": false_negative,
    }
    if subgroup_mask is not None:
        mask = np.asarray(subgroup_mask, dtype=bool)
        if len(mask) != len(truth):
            raise ValueError("subgroup mask must align with labels")
        if mask.any():
            subgroup = binary_classification_metrics(truth[mask], scores[mask], bins=bins)
            result.update({f"subgroup_{key}": value for key, value in subgroup.items() if key != "n_objects"})
            result["subgroup_n_objects"] = int(mask.sum())
    return result


def binary_auroc(labels: np.ndarray, scores: np.ndarray) -> float:
    positives = scores[labels == 1]
    negatives = scores[labels == 0]
    if not len(positives) or not len(negatives):
        return math.nan
    ordered = np.argsort(scores, kind="mergesort")
    ranks = np.empty_like(ordered, dtype=float)
    ranks[ordered] = np.arange(1, len(scores) + 1)
    positive_rank_sum = float(np.sum(ranks[labels == 1]))
    return (positive_rank_sum - len(positives) * (len(positives) + 1) / 2.0) / (len(positives) * len(negatives))


def expected_calibration_error(labels: np.ndarray, scores: np.ndarray, *, bins: int = 10) -> float:
    edges = np.linspace(0.0, 1.0, bins + 1)
    value = 0.0
    for lower, upper in zip(edges[:-1], edges[1:]):
        mask = (scores >= lower) & (scores < upper if upper < 1.0 else scores <= upper)
        if mask.any():
            value += float(mask.mean()) * abs(float(scores[mask].mean()) - float(labels[mask].mean()))
    return value


def decision_policy_metrics(
    labels: Sequence[int],
    predictions: Sequence[int],
    actions: Sequence[str],
    critical_mask: Sequence[bool],
    *,
    costs: dict[str, float],
) -> dict[str, float | int]:
    truth = np.asarray(labels, dtype=int)
    predicted = np.asarray(predictions, dtype=int)
    action = np.asarray(actions, dtype=str)
    critical = np.asarray(critical_mask, dtype=bool)
    if not (len(truth) == len(predicted) == len(action) == len(critical)):
        raise ValueError("policy vectors must align")
    automatic = action == "accept"
    wrong = predicted != truth
    wrong_auto = automatic & wrong
    critical_wrong = wrong_auto & critical
    review = action == "review"
    blocked = action == "block"
    false_block = blocked & ~wrong
    risk = (
        costs["critical_error"] * int(critical_wrong.sum())
        + costs["wrong_auto"] * int(wrong_auto.sum())
        + costs["review"] * int(review.sum())
        + costs["false_block"] * int(false_block.sum())
    )
    return {
        "n_objects": len(truth),
        "automatic_coverage": float(automatic.mean()),
        "correct_automatic": int((automatic & ~wrong).sum()),
        "wrong_automatic": int(wrong_auto.sum()),
        "critical_wrong_automatic": int(critical_wrong.sum()),
        "reviewed": int(review.sum()),
        "blocked": int(blocked.sum()),
        "false_blocks": int(false_block.sum()),
        "missed_critical_cases": int(critical_wrong.sum()),
        "unresolved": int(np.sum(~np.isin(action, ["accept", "review", "block"]))),
        "risk": float(risk),
        "mean_cost": float(risk / len(truth)),
    }


def _safe_ratio(numerator: float, denominator: float) -> float:
    return float(numerator / denominator) if denominator else 0.0
