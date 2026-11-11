"""Ensure validation-fold statistics cannot enter group-OOF controller training."""

from __future__ import annotations

import numpy as np
from fuzzyxai.selective_observer.controller import _oof_probabilities
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


def test_grouped_oof_fits_scaler_inside_each_training_fold() -> None:
    seed = 4201
    matrix = []
    labels = []
    groups = []
    for group in range(24):
        group_shift = 80.0 if group % 3 == 0 else (-60.0 if group % 3 == 1 else 5.0)
        for row in range(4):
            matrix.append([group_shift + row * 0.7, (group % 7) * 0.25 + row, 0.1 * group - row])
            labels.append(row % 2)
            groups.append(group)
    features = np.asarray(matrix, dtype=float)
    outcomes = np.asarray(labels, dtype=int)
    group_ids = np.asarray(groups)
    observed = _oof_probabilities(features, outcomes, group_ids, folds=5, seed=seed)

    expected = np.zeros(len(outcomes), dtype=float)
    splitter = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=seed)
    for train, validation in splitter.split(features, outcomes, group_ids):
        reference = make_pipeline(
            StandardScaler(),
            LogisticRegression(max_iter=1000, class_weight="balanced", random_state=seed),
        )
        reference.fit(features[train], outcomes[train])
        expected[validation] = reference.predict_proba(features[validation])[:, 1]

    np.testing.assert_allclose(observed, expected, rtol=1e-12, atol=1e-12)
    assert np.isfinite(observed).all()
    assert ((observed >= 0.0) & (observed <= 1.0)).all()
