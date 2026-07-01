from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Mapping, Sequence

import numpy as np


BETA_ORDER = ["repr", "rules", "activations", "uncertainty", "trace", "reduction"]


def _project_to_simplex(v: np.ndarray) -> np.ndarray:
    """Euclidean projection to the probability simplex."""
    v = np.asarray(v, dtype=float)
    if v.ndim != 1:
        raise ValueError("v must be one-dimensional")
    n = v.size
    u = np.sort(v)[::-1]
    cssv = np.cumsum(u)
    rho_candidates = u * np.arange(1, n + 1) > (cssv - 1)
    if not np.any(rho_candidates):
        return np.ones(n) / n
    rho = np.nonzero(rho_candidates)[0][-1]
    theta = (cssv[rho] - 1) / (rho + 1)
    return np.maximum(v - theta, 0)


def predict_disagreement(features: Mapping[str, float], beta: Mapping[str, float]) -> float:
    """Predict expert disagreement from component distances and beta weights."""
    return float(sum(float(beta.get(k, 0.0)) * float(features.get(k, 0.0)) for k in BETA_ORDER))


def calibrate_beta_weights(
    feature_rows: Sequence[Mapping[str, float]],
    expert_scores: Sequence[float],
    *,
    l2: float = 1e-3,
    steps: int = 2000,
    lr: float = 0.05,
    include_reduction: bool = True,
) -> Dict[str, float]:
    """Calibrate beta weights on the simplex from expert disagreement labels.

    This implements the reproducibility layer for chapter 2: beta is no longer
    arbitrary. Given component distances and expert scores r in [0,1], the
    function solves a regularized least-squares problem using projected gradient
    descent on the simplex.
    """
    keys = BETA_ORDER if include_reduction else BETA_ORDER[:-1]
    X = np.asarray([[float(row.get(k, 0.0)) for k in keys] for row in feature_rows], dtype=float)
    y = np.asarray([float(v) for v in expert_scores], dtype=float)
    if X.ndim != 2 or len(X) != len(y):
        raise ValueError("feature_rows and expert_scores are inconsistent")
    if len(y) == 0:
        raise ValueError("at least one calibration pair is required")
    w = np.ones(len(keys), dtype=float) / len(keys)
    for _ in range(steps):
        pred = X @ w
        grad = (2.0 / len(y)) * (X.T @ (pred - y)) + 2.0 * l2 * w
        w = _project_to_simplex(w - lr * grad)
    return {k: float(v) for k, v in zip(keys, w)}
