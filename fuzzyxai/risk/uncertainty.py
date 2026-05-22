from __future__ import annotations

import numpy as np


def _as_2d(proba) -> np.ndarray:
    arr = np.asarray(proba, dtype=float)
    if arr.ndim == 1:
        arr = arr.reshape(1, -1)
    if arr.ndim != 2:
        raise ValueError('proba must have shape (n_samples, n_classes)')
    return np.clip(arr, 0.0, 1.0)


def entropy_uncertainty(proba, eps: float = 1e-12) -> np.ndarray:
    """Normalized entropy uncertainty in [0, 1]."""
    arr = _as_2d(proba)
    n_classes = arr.shape[1]
    if n_classes <= 1:
        return np.zeros(arr.shape[0], dtype=float)
    row_sum = arr.sum(axis=1, keepdims=True)
    safe = np.divide(arr, np.maximum(row_sum, eps))
    entropy = -(safe * np.log(np.maximum(safe, eps))).sum(axis=1)
    return np.clip(entropy / np.log(n_classes), 0.0, 1.0)


def margin_uncertainty(proba) -> np.ndarray:
    """Uncertainty from the gap between the two largest probabilities."""
    arr = _as_2d(proba)
    if arr.shape[1] <= 1:
        return np.zeros(arr.shape[0], dtype=float)
    sorted_probs = np.sort(arr, axis=1)
    margin = sorted_probs[:, -1] - sorted_probs[:, -2]
    return np.clip(1.0 - margin, 0.0, 1.0)


def confidence_from_uncertainty(uncertainty) -> np.ndarray:
    return np.clip(1.0 - np.asarray(uncertainty, dtype=float), 0.0, 1.0)


def ensemble_variance(probas) -> np.ndarray:
    """Optional ensemble uncertainty from per-estimator probabilities.

    Expected shape: (n_estimators, n_samples, n_classes).
    """
    arr = np.asarray(probas, dtype=float)
    if arr.ndim != 3:
        raise ValueError('probas must have shape (n_estimators, n_samples, n_classes)')
    class_var = arr.var(axis=0).mean(axis=1)
    return np.clip(class_var / 0.25, 0.0, 1.0)
