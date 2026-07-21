"""Sensitivity and action-robustness calculations for calibrated policies."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np

from .metrics import decision_policy_metrics
from .policies import PolicySignals, apply_policy


SCALE_FACTORS = (0.50, 0.75, 1.00, 1.25, 1.50)


@dataclass(frozen=True)
class SensitivityPoint:
    parameter: str
    factor: float
    value: float
    changed_action_fraction: float
    accept_fraction: float
    review_fraction: float
    block_fraction: float
    false_blocks: int
    missed_critical_cases: int
    mean_cost: float


def policy_sensitivity(
    *,
    labels: Sequence[int],
    predictions: Sequence[int],
    critical_mask: Sequence[bool],
    signals: PolicySignals,
    costs: dict[str, float],
    confidence_threshold: float,
    conflict_threshold: float,
    stability_threshold: float,
) -> tuple[list[SensitivityPoint], np.ndarray]:
    baseline = apply_policy(
        "P5",
        signals,
        confidence_threshold=confidence_threshold,
        conflict_threshold=conflict_threshold,
        stability_threshold=stability_threshold,
    )
    action_matrix: list[np.ndarray] = []
    points: list[SensitivityPoint] = []
    parameters = {
        "confidence_threshold": confidence_threshold,
        "conflict_threshold": conflict_threshold,
        "stability_threshold": stability_threshold,
    }
    for parameter, base_value in parameters.items():
        for factor in SCALE_FACTORS:
            values = dict(parameters)
            values[parameter] = float(np.clip(base_value * factor, 0.0, 1.0))
            actions = apply_policy(
                "P5",
                signals,
                confidence_threshold=values["confidence_threshold"],
                conflict_threshold=values["conflict_threshold"],
                stability_threshold=values["stability_threshold"],
            )
            metrics = decision_policy_metrics(labels, predictions, actions, critical_mask, costs=costs)
            action_matrix.append(actions)
            points.append(
                SensitivityPoint(
                    parameter=parameter,
                    factor=factor,
                    value=values[parameter],
                    changed_action_fraction=float(np.mean(actions != baseline)),
                    accept_fraction=float(np.mean(actions == "accept")),
                    review_fraction=float(np.mean(actions == "review")),
                    block_fraction=float(np.mean(actions == "block")),
                    false_blocks=int(metrics["false_blocks"]),
                    missed_critical_cases=int(metrics["missed_critical_cases"]),
                    mean_cost=float(metrics["mean_cost"]),
                )
            )
    stacked = np.vstack(action_matrix)
    robustness = np.mean(stacked == baseline[None, :], axis=0)
    return points, robustness


def perturbation_scenarios(values: np.ndarray, *, seed: int = 42) -> dict[str, np.ndarray]:
    matrix = np.asarray(values, dtype=float)
    rng = np.random.default_rng(seed)
    noisy = matrix + rng.normal(0.0, 0.05 * np.maximum(matrix.std(axis=0), 1e-6), size=matrix.shape)
    missing = matrix.copy()
    missing_mask = rng.random(matrix.shape) < 0.05
    medians = np.nanmedian(matrix, axis=0)
    missing[missing_mask] = np.broadcast_to(medians, matrix.shape)[missing_mask]
    shifted = matrix + 0.2 * np.maximum(matrix.std(axis=0), 1e-6)
    background_shift = matrix - np.mean(matrix, axis=0) + np.median(matrix, axis=0)
    return {
        "feature_noise": noisy,
        "missing_values_median_imputed": missing,
        "distribution_shift": shifted,
        "background_reference_shift": background_shift,
    }
