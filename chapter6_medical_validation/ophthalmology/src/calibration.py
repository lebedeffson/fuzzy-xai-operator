from __future__ import annotations

from typing import cast

import numpy as np


def probabilities_from_logits(logits: np.ndarray, temperature: float = 1.0) -> np.ndarray:
    values = np.asarray(logits, dtype=float)
    if values.ndim != 2 or not np.isfinite(values).all():
        raise ValueError("logits must be a finite N x C matrix")
    if not np.isfinite(temperature) or temperature <= 0:
        raise ValueError("temperature must be finite and positive")
    scaled = values / float(temperature)
    scaled -= scaled.max(axis=1, keepdims=True)
    exponential = np.exp(scaled)
    return cast(np.ndarray, exponential / exponential.sum(axis=1, keepdims=True))


def temperature_nll(logits: np.ndarray, labels: np.ndarray, temperature: float) -> float:
    probabilities = probabilities_from_logits(logits, temperature)
    truth = np.asarray(labels, dtype=int)
    if truth.shape != (len(probabilities),) or np.any((truth < 0) | (truth >= probabilities.shape[1])):
        raise ValueError("labels do not match logits")
    return float(-np.log(np.clip(probabilities[np.arange(len(truth)), truth], 1e-15, 1.0)).mean())


def fit_temperature(logits: np.ndarray, labels: np.ndarray, *, lower: float = 0.05, upper: float = 10.0, iterations: int = 100) -> dict[str, float | int | str]:
    """Fit one scalar on the validation split with deterministic golden-section search."""

    phi = (1 + np.sqrt(5.0)) / 2.0
    left, right = float(lower), float(upper)
    x1, x2 = right - (right - left) / phi, left + (right - left) / phi
    f1, f2 = temperature_nll(logits, labels, x1), temperature_nll(logits, labels, x2)
    for _ in range(iterations):
        if f1 <= f2:
            right, x2, f2 = x2, x1, f1
            x1 = right - (right - left) / phi
            f1 = temperature_nll(logits, labels, x1)
        else:
            left, x1, f1 = x1, x2, f2
            x2 = left + (right - left) / phi
            f2 = temperature_nll(logits, labels, x2)
    temperature = (left + right) / 2.0
    return {"status": "fitted_on_validation_only", "temperature": temperature, "uncalibrated_nll": temperature_nll(logits, labels, 1.0), "calibrated_nll": temperature_nll(logits, labels, temperature), "iterations": iterations}
