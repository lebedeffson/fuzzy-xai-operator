"""OOF risk-head fitting and frozen calibration helpers."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from typing import Mapping, Sequence

import numpy as np


@dataclass(frozen=True)
class RiskHeadTrainingRow:
    object_id: str
    group_id: str
    features: Mapping[str, float]
    target: bool
    source_features_are_oof: bool = True
    partition: str = "development"

    def __post_init__(self) -> None:
        if self.partition == "test":
            raise ValueError("test rows cannot be used to fit a risk head")
        if not self.source_features_are_oof:
            raise ValueError("risk-head source features must be out-of-fold")


@dataclass(frozen=True)
class CalibratedRiskHead:
    target_name: str
    feature_names: tuple[str, ...]
    coefficients: tuple[float, ...]
    intercept: float
    means: tuple[float, ...]
    scales: tuple[float, ...]
    calibration_method: str
    calibration_parameters: tuple[float, ...]
    modality: str
    development_sha256: str

    def predict(self, features: Mapping[str, float]) -> float:
        missing = set(self.feature_names) - set(features)
        if missing:
            raise ValueError(f"missing risk-head features: {sorted(missing)}")
        values = np.asarray([features[name] for name in self.feature_names], dtype=float)
        standardized = (values - np.asarray(self.means)) / np.asarray(self.scales)
        raw = _sigmoid(float(np.dot(standardized, np.asarray(self.coefficients)) + self.intercept))
        return apply_calibration(raw, self.calibration_method, self.calibration_parameters)


def fit_risk_head_oof(
    rows: Sequence[RiskHeadTrainingRow],
    *,
    target_name: str,
    feature_names: Sequence[str],
    modality: str = "global",
    calibration_method: str = "platt",
    folds: int = 5,
    seed: int = 4201,
) -> tuple[CalibratedRiskHead, tuple[float, ...]]:
    if len(rows) < max(20, 4 * folds):
        raise ValueError("at least 20 development rows are required")
    if any(row.partition == "test" or not row.source_features_are_oof for row in rows):
        raise ValueError("risk heads may use OOF development rows only")
    names = tuple(feature_names)
    matrix = np.asarray([[row.features[name] for name in names] for row in rows], dtype=float)
    labels = np.asarray([row.target for row in rows], dtype=int)
    groups = np.asarray([row.group_id for row in rows])
    if len(np.unique(labels)) != 2:
        raise ValueError("both target classes are required")
    means = matrix.mean(axis=0)
    scales = matrix.std(axis=0)
    scales[scales < 1e-12] = 1.0
    standardized = (matrix - means) / scales
    oof_raw = _oof_scores(standardized, labels, groups, folds=folds, seed=seed)
    calibration_parameters = fit_calibration(oof_raw, labels, calibration_method)
    model = _fit_logistic(standardized, labels, seed)
    digest_payload = [
        {"object_id": row.object_id, "group_id": row.group_id, "target": row.target, "features": [row.features[name] for name in names]}
        for row in rows
    ]
    digest = hashlib.sha256(json.dumps(digest_payload, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    head = CalibratedRiskHead(
        target_name=target_name,
        feature_names=names,
        coefficients=tuple(float(value) for value in model.coef_[0]),
        intercept=float(model.intercept_[0]),
        means=tuple(float(value) for value in means),
        scales=tuple(float(value) for value in scales),
        calibration_method=calibration_method,
        calibration_parameters=calibration_parameters,
        modality=modality,
        development_sha256=digest,
    )
    return head, tuple(apply_calibration(value, calibration_method, calibration_parameters) for value in oof_raw)


def fit_calibration(scores: np.ndarray, labels: np.ndarray, method: str) -> tuple[float, ...]:
    if method == "platt":
        clipped = np.clip(scores, 1e-6, 1 - 1e-6)
        logits = np.log(clipped / (1 - clipped)).reshape(-1, 1)
        model = _fit_logistic(logits, labels, 4201)
        return (float(model.coef_[0][0]), float(model.intercept_[0]))
    if method == "temperature":
        candidates = np.geomspace(0.25, 4.0, 81)
        clipped = np.clip(scores, 1e-6, 1 - 1e-6)
        logits = np.log(clipped / (1 - clipped))
        losses = [float(np.mean(-(labels * np.log(np.clip(_sigmoid_array(logits / t), 1e-9, 1)) + (1 - labels) * np.log(np.clip(1 - _sigmoid_array(logits / t), 1e-9, 1))))) for t in candidates]
        return (float(candidates[int(np.argmin(losses))]),)
    if method == "identity":
        return (1.0,)
    raise ValueError("supported calibration methods are platt, temperature, and identity")


def apply_calibration(score: float, method: str, parameters: Sequence[float]) -> float:
    score = float(np.clip(score, 1e-9, 1 - 1e-9))
    if method == "platt":
        logit = np.log(score / (1 - score))
        return _sigmoid(float(parameters[0] * logit + parameters[1]))
    if method == "temperature":
        logit = np.log(score / (1 - score))
        return _sigmoid(float(logit / parameters[0]))
    if method == "identity":
        return score
    raise ValueError("unknown calibration method")


def _fit_logistic(matrix: np.ndarray, labels: np.ndarray, seed: int):
    from sklearn.linear_model import LogisticRegression

    model = LogisticRegression(max_iter=1000, class_weight="balanced", random_state=seed)
    model.fit(matrix, labels)
    return model


def _oof_scores(matrix: np.ndarray, labels: np.ndarray, groups: np.ndarray, *, folds: int, seed: int) -> np.ndarray:
    from sklearn.model_selection import StratifiedGroupKFold

    effective = min(folds, int(np.bincount(labels).min()), len(np.unique(groups)))
    if effective < 2:
        raise ValueError("at least two OOF folds are required")
    result = np.zeros(len(labels), dtype=float)
    splitter = StratifiedGroupKFold(n_splits=effective, shuffle=True, random_state=seed)
    for train, validation in splitter.split(matrix, labels, groups):
        result[validation] = _fit_logistic(matrix[train], labels[train], seed).predict_proba(matrix[validation])[:, 1]
    return result


def _sigmoid(value: float) -> float:
    return float(1.0 / (1.0 + np.exp(-np.clip(value, -40.0, 40.0))))


def _sigmoid_array(values: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(values, -40.0, 40.0)))
