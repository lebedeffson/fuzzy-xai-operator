"""Non-refit rule necessity for the current fitted model."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

import numpy as np


@dataclass(frozen=True)
class RuleEffectEstimate:
    estimand: str
    baseline_metric: float
    ablated_metric: float
    effect: float
    n_objects: int
    metadata: dict[str, Any]


def nonrefit_effect(
    model: object,
    rule: object,
    values: np.ndarray,
    labels: np.ndarray,
    *,
    metric: Callable[[np.ndarray, np.ndarray], float],
) -> RuleEffectEstimate:
    if not hasattr(model, "predict_without_rule"):
        raise TypeError("non-refit ablation requires model.predict_without_rule(values, rule)")
    baseline = np.asarray(model.predict(values))  # type: ignore[attr-defined]
    ablated = np.asarray(model.predict_without_rule(values, rule))  # type: ignore[attr-defined]
    baseline_metric = float(metric(labels, baseline))
    ablated_metric = float(metric(labels, ablated))
    return RuleEffectEstimate("nonrefit", baseline_metric, ablated_metric, baseline_metric - ablated_metric, len(labels), {"refit": False})
