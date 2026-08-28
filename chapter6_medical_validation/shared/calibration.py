from __future__ import annotations

import numpy as np


def softmax(logits: np.ndarray, temperature: float = 1.0) -> np.ndarray:
    value = np.asarray(logits, dtype=float) / float(temperature)
    value -= value.max(axis=1, keepdims=True)
    exp = np.exp(value)
    return exp / exp.sum(axis=1, keepdims=True)


def fit_temperature(logits: np.ndarray, labels: np.ndarray, *, iterations: int = 100) -> dict[str, float | str]:
    value, truth = np.asarray(logits, dtype=float), np.asarray(labels, dtype=int)
    def nll(temperature: float) -> float:
        prob = softmax(value, temperature)
        return float(-np.log(np.clip(prob[np.arange(len(truth)), truth], 1e-15, 1)).mean())
    phi, left, right = (1 + np.sqrt(5)) / 2, 0.05, 10.0
    x1, x2 = right - (right - left) / phi, left + (right - left) / phi
    f1, f2 = nll(x1), nll(x2)
    for _ in range(iterations):
        if f1 <= f2:
            right, x2, f2 = x2, x1, f1; x1 = right - (right - left) / phi; f1 = nll(x1)
        else:
            left, x1, f1 = x1, x2, f2; x2 = left + (right - left) / phi; f2 = nll(x2)
    temperature = float((left + right) / 2)
    return {"status": "fitted_validation_only", "temperature": temperature, "uncalibrated_nll": nll(1.0), "calibrated_nll": nll(temperature)}
