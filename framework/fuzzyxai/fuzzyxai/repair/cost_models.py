from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from math import log1p
from typing import Literal

from .baselines import RepairAction

CostModelId = Literal["uniform", "runtime", "dependency_weighted", "hybrid"]


@dataclass(frozen=True)
class CostCalibration:
    runtime_ms_by_kind: dict[str, float]
    runtime_scale_ms: float
    fitted_split: str = "development"

    def normalized_runtime(self, action: RepairAction) -> float:
        measured = self.runtime_ms_by_kind.get(action.action_kind)
        if measured is None:
            raise KeyError(f"action kind was not calibrated: {action.action_kind}")
        return measured / max(self.runtime_scale_ms, 1e-12)


@dataclass(frozen=True)
class CostWeights:
    alpha: float = 0.5
    beta: float = 0.5
    gamma: float = 0.5


DEFAULT_COST_WEIGHTS = CostWeights()


def action_cost(
    action: RepairAction,
    model: CostModelId,
    calibration: CostCalibration,
    weights: CostWeights = DEFAULT_COST_WEIGHTS,
) -> float:
    if model == "uniform":
        return 1.0
    if model == "runtime":
        return calibration.normalized_runtime(action)
    dependency = 1.0 + log1p(action.dependency_fanout)
    if model == "dependency_weighted":
        return dependency
    if model == "hybrid":
        return (
            1.0
            + weights.alpha * calibration.normalized_runtime(action)
            + weights.beta * log1p(action.dependency_fanout)
            + weights.gamma * action.rollback_risk
        )
    raise ValueError(f"unsupported cost model: {model}")


def plan_cost(
    actions: Iterable[RepairAction],
    model: CostModelId,
    calibration: CostCalibration,
    weights: CostWeights = DEFAULT_COST_WEIGHTS,
) -> float:
    return sum(action_cost(action, model, calibration, weights) for action in actions)
