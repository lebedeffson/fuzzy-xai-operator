from __future__ import annotations

import sys
import types

import numpy as np

from fuzzyxai.q1_final.explainers import _rulefit_values


class _ThreeClassModel:
    def predict(self, values: np.ndarray) -> np.ndarray:
        return np.argmax(values[:, :3], axis=1)


def test_rulefit_uses_one_vs_rest_for_multiclass(monkeypatch: object) -> None:
    class FakeRuleFitClassifier:
        fitted = 0

        def __init__(self, **_: object) -> None:
            self.class_index = FakeRuleFitClassifier.fitted
            FakeRuleFitClassifier.fitted += 1
            self.classes_ = np.asarray([0, 1])

        def fit(self, values: np.ndarray, target: np.ndarray) -> "FakeRuleFitClassifier":
            assert set(np.unique(target)) <= {0, 1}
            self.feature_importances_ = np.full(values.shape[1], self.class_index + 1.0)
            return self

        def predict_proba(self, values: np.ndarray) -> np.ndarray:
            positive = np.full(len(values), 0.2 + 0.1 * self.class_index)
            return np.column_stack((1.0 - positive, positive))

    monkeypatch.setitem(sys.modules, "imodels", types.SimpleNamespace(RuleFitClassifier=FakeRuleFitClassifier))
    train = np.asarray(
        [[3.0, 1.0, 0.0], [0.0, 3.0, 1.0], [1.0, 0.0, 3.0]],
        dtype=float,
    )
    sample = np.asarray([[4.0, 1.0, 0.0], [0.0, 4.0, 1.0]], dtype=float)

    values, fidelity = _rulefit_values(_ThreeClassModel(), train, np.asarray([0, 1, 2]), sample)

    assert FakeRuleFitClassifier.fitted == 3
    assert values.shape == sample.shape
    assert fidelity.shape == (2,)
    assert np.all(np.isfinite(fidelity))
    assert np.all((0.0 <= fidelity) & (fidelity <= 1.0))
