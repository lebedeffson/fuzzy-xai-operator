"""Paired effect, interval and multiplicity helpers for preregistered endpoints."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np


def paired_bootstrap_difference(
    candidate: Sequence[float],
    baseline: Sequence[float],
    *,
    repetitions: int = 4000,
    seed: int = 4201,
) -> dict[str, object]:
    left, right = _paired(candidate, baseline)
    differences = left - right
    rng = np.random.default_rng(seed)
    samples = np.empty(repetitions, dtype=float)
    for index in range(repetitions):
        selected = rng.integers(0, len(differences), size=len(differences))
        samples[index] = float(np.mean(differences[selected]))
    return {
        "n": len(differences),
        "effect": float(np.mean(differences)),
        "confidence_interval_95": [float(np.quantile(samples, 0.025)), float(np.quantile(samples, 0.975))],
        "unit_of_analysis": "paired replicate",
    }


def paired_permutation_pvalue(
    candidate: Sequence[float],
    baseline: Sequence[float],
    *,
    repetitions: int = 10000,
    seed: int = 4201,
) -> float:
    left, right = _paired(candidate, baseline)
    differences = left - right
    observed = abs(float(np.mean(differences)))
    rng = np.random.default_rng(seed)
    exceed = 1
    for _ in range(repetitions):
        signs = rng.choice((-1.0, 1.0), size=len(differences))
        exceed += int(abs(float(np.mean(differences * signs))) >= observed)
    return exceed / (repetitions + 1)


def holm_adjust(p_values: Sequence[float]) -> list[float]:
    values = np.asarray(p_values, dtype=float)
    if values.ndim != 1 or len(values) == 0 or np.any((values < 0.0) | (values > 1.0)):
        raise ValueError("p-values must be a non-empty one-dimensional sequence in [0, 1]")
    order = np.argsort(values)
    adjusted = np.empty(len(values), dtype=float)
    running = 0.0
    for rank, index in enumerate(order):
        current = min(1.0, (len(values) - rank) * float(values[index]))
        running = max(running, current)
        adjusted[index] = running
    return adjusted.tolist()


def _paired(candidate: Sequence[float], baseline: Sequence[float]) -> tuple[np.ndarray, np.ndarray]:
    left = np.asarray(candidate, dtype=float)
    right = np.asarray(baseline, dtype=float)
    if left.ndim != 1 or right.ndim != 1 or len(left) != len(right) or len(left) < 2:
        raise ValueError("paired inputs must be aligned one-dimensional samples with at least two values")
    if not np.all(np.isfinite(left)) or not np.all(np.isfinite(right)):
        raise ValueError("paired inputs must be finite")
    return left, right
