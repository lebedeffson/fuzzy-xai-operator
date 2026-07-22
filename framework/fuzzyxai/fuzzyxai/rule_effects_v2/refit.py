"""Refit rule replaceability estimand."""

from __future__ import annotations

from typing import Callable

import numpy as np

from .nonrefit import RuleEffectEstimate


def refit_effect(
    model: object,
    rule: object,
    train_values: np.ndarray,
    train_labels: np.ndarray,
    test_values: np.ndarray,
    test_labels: np.ndarray,
    *,
    metric: Callable[[np.ndarray, np.ndarray], float],
) -> RuleEffectEstimate:
    if not hasattr(model, "refit_without_rule"):
        raise TypeError("refit ablation requires model.refit_without_rule(rule, train_values, train_labels)")
    baseline = np.asarray(model.predict(test_values))  # type: ignore[attr-defined]
    ablated_model = model.refit_without_rule(rule, train_values, train_labels)  # type: ignore[attr-defined]
    ablated = np.asarray(ablated_model.predict(test_values))
    baseline_metric = float(metric(test_labels, baseline))
    ablated_metric = float(metric(test_labels, ablated))
    return RuleEffectEstimate("refit", baseline_metric, ablated_metric, baseline_metric - ablated_metric, len(test_labels), {"refit": True})
