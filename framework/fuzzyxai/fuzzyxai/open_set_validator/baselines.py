"""Open-set fault baselines evaluated on the same structural vectors."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np


@dataclass(frozen=True)
class AnomalyScore:
    method: str
    score: float


def maximum_softmax_unknown(probabilities: Sequence[float]) -> AnomalyScore:
    if not probabilities:
        raise ValueError("known-family probabilities are required")
    return AnomalyScore("maximum_softmax_probability", 1.0 - max(float(value) for value in probabilities))


def mahalanobis_unknown(values: np.ndarray, mean: np.ndarray, covariance: np.ndarray) -> AnomalyScore:
    delta = np.asarray(values, dtype=float) - np.asarray(mean, dtype=float)
    inverse = np.linalg.pinv(np.asarray(covariance, dtype=float))
    return AnomalyScore("mahalanobis_distance", float(np.sqrt(max(0.0, delta @ inverse @ delta))))


def graph_anomaly_score(*, unseen_edges: int, total_edges: int, unexpected_depth: float, type_mismatches: int) -> AnomalyScore:
    if total_edges < 0 or unseen_edges < 0 or type_mismatches < 0:
        raise ValueError("graph counts cannot be negative")
    edge_ratio = unseen_edges / max(1, total_edges)
    score = 0.5 * edge_ratio + 0.3 * min(1.0, unexpected_depth) + 0.2 * min(1.0, type_mismatches / 3.0)
    return AnomalyScore("graph_anomaly_score", float(score))


def fit_one_class_scores(train: np.ndarray, test: np.ndarray, *, seed: int = 4201) -> np.ndarray:
    from sklearn.ensemble import IsolationForest

    model = IsolationForest(n_estimators=100, contamination="auto", random_state=seed, n_jobs=1)
    model.fit(train)
    return np.asarray(-model.score_samples(test), dtype=float)
