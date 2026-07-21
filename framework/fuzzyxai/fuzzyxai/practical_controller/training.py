"""Fit a frozen practical policy only from OOF development records."""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict
from typing import Sequence

import numpy as np

from .calibration import compare_calibrators
from .contracts import PracticalDevelopmentExample, PracticalPolicy


def fit_practical_policy(
    examples: Sequence[PracticalDevelopmentExample],
    *,
    policy_version: str,
    seed: int = 4201,
) -> tuple[PracticalPolicy, dict[str, object]]:
    if len(examples) < 40:
        raise ValueError("practical policy development requires at least 40 OOF examples")
    predictive = np.asarray([item.predictive_features for item in examples], dtype=float)
    route = np.asarray([item.route_features for item in examples], dtype=float)
    labels = np.asarray([item.operationally_invalid_action for item in examples], dtype=int)
    groups = np.asarray([item.group_id for item in examples])
    if set(np.unique(labels)) != {0, 1}:
        raise ValueError("both valid and invalid development outcomes are required")
    predictive_oof, predictive_model = _fit_oof(predictive, labels, groups, seed=seed)
    route_oof, route_model = _fit_oof(route, labels, groups, seed=seed + 1)
    calibration = compare_calibrators(predictive_oof, labels, seed=seed)
    combined = 1.0 - (1.0 - predictive_oof) * (1.0 - route_oof)
    accept, short, full = (float(np.quantile(combined, quantile)) for quantile in (0.70, 0.85, 0.97))
    development_hash = hashlib.sha256(
        json.dumps([asdict(item) for item in examples], sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    policy = PracticalPolicy(
        schema_version="1.0",
        policy_version=policy_version,
        predictive_weights=tuple(float(value) for value in predictive_model.coef_[0]),
        predictive_intercept=float(predictive_model.intercept_[0]),
        route_weights=tuple(float(value) for value in route_model.coef_[0]),
        route_intercept=float(route_model.intercept_[0]),
        accept_max_risk=accept,
        short_review_max_risk=short,
        full_review_max_risk=full,
        calibration_method=str(calibration["selected_method"]),
        calibration_parameters=tuple(float(value) for value in calibration["selected_parameters"]),
        development_sha256=development_hash,
        selected_without_test=True,
    )
    return policy, {
        "phase": "formative_development",
        "n_examples": len(examples),
        "source_features_are_oof": True,
        "confirmatory_test_used": False,
        "calibration": calibration,
        "thresholds": {"accept": accept, "short_review": short, "full_review": full},
        "development_sha256": development_hash,
    }


def _fit_oof(matrix: np.ndarray, labels: np.ndarray, groups: np.ndarray, *, seed: int):
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import StratifiedGroupKFold

    folds = min(5, int(np.bincount(labels).min()), len(np.unique(groups)))
    if folds < 2:
        raise ValueError("OOF fitting requires at least two groups per class")
    splitter = StratifiedGroupKFold(n_splits=folds, shuffle=True, random_state=seed)
    oof = np.zeros(len(labels), dtype=float)
    for train, validation in splitter.split(matrix, labels, groups):
        model = LogisticRegression(max_iter=1000, class_weight="balanced", random_state=seed).fit(matrix[train], labels[train])
        oof[validation] = model.predict_proba(matrix[validation])[:, 1]
    final = LogisticRegression(max_iter=1000, class_weight="balanced", random_state=seed).fit(matrix, labels)
    return oof, final
