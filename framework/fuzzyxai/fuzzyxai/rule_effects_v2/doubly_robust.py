"""Cross-fitted doubly robust rule-effect estimator."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class DoublyRobustEffect:
    effect: float
    standard_error: float
    confidence_interval_95: tuple[float, float]
    folds: int
    n_objects: int
    propensity_clipped_fraction: float
    source_predictions_oof: bool


def cross_fitted_doubly_robust_effect(
    covariates: np.ndarray,
    treatment: np.ndarray,
    outcome: np.ndarray,
    *,
    groups: np.ndarray | None = None,
    folds: int = 5,
    seed: int = 4201,
    propensity_clip: float = 0.05,
) -> DoublyRobustEffect:
    from sklearn.ensemble import HistGradientBoostingRegressor
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import GroupKFold, StratifiedKFold

    values = np.asarray(covariates, dtype=float)
    treated = np.asarray(treatment, dtype=int)
    result = np.asarray(outcome, dtype=float)
    if len(values) != len(treated) or len(values) != len(result) or len(values) < 40:
        raise ValueError("aligned covariates, treatment and outcome with at least 40 rows are required")
    if set(np.unique(treated)) != {0, 1}:
        raise ValueError("treatment must contain both activation states")
    propensity = np.zeros(len(values), dtype=float)
    mu0 = np.zeros(len(values), dtype=float)
    mu1 = np.zeros(len(values), dtype=float)
    if groups is None:
        splitter = StratifiedKFold(n_splits=folds, shuffle=True, random_state=seed)
        splits = splitter.split(values, treated)
    else:
        splitter = GroupKFold(n_splits=folds)
        splits = splitter.split(values, treated, np.asarray(groups))
    for train, validation in splits:
        propensity_model = LogisticRegression(max_iter=1000, class_weight="balanced", random_state=seed)
        propensity_model.fit(values[train], treated[train])
        propensity[validation] = propensity_model.predict_proba(values[validation])[:, 1]
        for treatment_value, target in ((0, mu0), (1, mu1)):
            subset = train[treated[train] == treatment_value]
            if len(subset) < 10:
                raise ValueError("each fold requires at least ten rows per treatment state")
            model = HistGradientBoostingRegressor(max_iter=80, max_leaf_nodes=15, random_state=seed)
            model.fit(values[subset], result[subset])
            target[validation] = model.predict(values[validation])
    clipped = np.clip(propensity, propensity_clip, 1.0 - propensity_clip)
    influence = mu1 - mu0 + treated * (result - mu1) / clipped - (1 - treated) * (result - mu0) / (1.0 - clipped)
    effect = float(np.mean(influence))
    standard_error = float(np.std(influence, ddof=1) / np.sqrt(len(influence)))
    return DoublyRobustEffect(
        effect,
        standard_error,
        (effect - 1.96 * standard_error, effect + 1.96 * standard_error),
        folds,
        len(values),
        float(np.mean(propensity != clipped)),
        True,
    )
