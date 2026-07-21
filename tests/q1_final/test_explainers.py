from __future__ import annotations

import sys
import types

import numpy as np

from fuzzyxai.q1_final.explainers import _rulefit_local_values, _rulefit_values


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
            self.include_linear = True
            self.lin_standardise = False
            self.coef = np.full(values.shape[1], self.class_index + 1.0)
            self.rules_without_feature_names_ = ()
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
    assert np.array_equal(values[0], sample[0])
    assert np.array_equal(values[1], 2.0 * sample[1])


def test_rulefit_local_values_distribute_active_rule_coefficient() -> None:
    class Rule:
        agg_dict = {("X_0", ">="): "0.0", ("X_2", "<="): "1.0"}
        args = [2.0]

        def __str__(self) -> str:
            return "X_0 >= 0.0 and X_2 <= 1.0"

    class Surrogate:
        include_linear = True
        lin_standardise = False
        coef = np.zeros(3)
        rules_without_feature_names_ = (Rule(),)

        def transform(self, sample: np.ndarray, _rules: list[str]) -> np.ndarray:
            return np.asarray([[1.0], [0.0]])

    sample = np.asarray([[1.0, 2.0, 0.5], [-1.0, 2.0, 2.0]])
    values = _rulefit_local_values(Surrogate(), sample, 3)
    assert np.array_equal(values[0], np.asarray([1.0, 0.0, 1.0]))
    assert np.array_equal(values[1], np.zeros(3))
