from __future__ import annotations

from collections.abc import Sequence
from typing import Mapping

import numpy as np

from .decision_policy import RiskAction


def _action_value(action) -> str:
    return action.value if isinstance(action, RiskAction) else str(action)


def accuracy_before(y_true, y_pred) -> float:
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    return float((y_true == y_pred).mean()) if len(y_true) else 0.0


def accepted_mask(actions: Sequence) -> np.ndarray:
    return np.asarray([_action_value(action) == RiskAction.ACCEPT.value for action in actions], dtype=bool)


def accepted_accuracy(y_true, y_pred, actions: Sequence) -> float | None:
    mask = accepted_mask(actions)
    if not mask.any():
        return None
    return accuracy_before(np.asarray(y_true)[mask], np.asarray(y_pred)[mask])


def coverage(actions: Sequence) -> float:
    if not actions:
        return 0.0
    return float(accepted_mask(actions).mean())


def defer_rate(actions: Sequence) -> float:
    if not actions:
        return 0.0
    values = [_action_value(action) for action in actions]
    return float(sum(v == RiskAction.DEFER_TO_HUMAN.value for v in values) / len(values))


def block_rate(actions: Sequence) -> float:
    if not actions:
        return 0.0
    values = [_action_value(action) for action in actions]
    return float(sum(v == RiskAction.BLOCK.value for v in values) / len(values))


def request_rate(actions: Sequence) -> float:
    if not actions:
        return 0.0
    values = [_action_value(action) for action in actions]
    return float(sum(v == RiskAction.REQUEST_MORE_DATA.value for v in values) / len(values))


def _cost(cost_matrix, true_label, pred_label) -> float:
    if isinstance(cost_matrix, Mapping):
        return float(cost_matrix.get((int(true_label), int(pred_label)), cost_matrix.get(f'{int(true_label)}->{int(pred_label)}', 0.0)))
    return float(np.asarray(cost_matrix, dtype=float)[int(true_label), int(pred_label)])


def expected_cost_before(y_true, y_pred, cost_matrix) -> float:
    if len(y_true) == 0:
        return 0.0
    return float(np.mean([_cost(cost_matrix, yt, yp) for yt, yp in zip(y_true, y_pred)]))


def expected_cost_after(
    y_true,
    y_pred,
    actions: Sequence,
    cost_matrix,
    defer_cost: float = 0.10,
    request_cost: float = 0.06,
    block_cost: float = 0.08,
) -> float:
    if len(y_true) == 0:
        return 0.0
    costs = []
    for yt, yp, action in zip(y_true, y_pred, actions):
        value = _action_value(action)
        if value in {RiskAction.ACCEPT.value, RiskAction.LOWER_CONFIDENCE.value}:
            costs.append(_cost(cost_matrix, yt, yp))
        elif value == RiskAction.REQUEST_MORE_DATA.value:
            costs.append(float(request_cost))
        elif value == RiskAction.BLOCK.value:
            costs.append(float(block_cost))
        else:
            costs.append(float(defer_cost))
    return float(np.mean(costs))


def risk_reduction(cost_before: float, cost_after: float) -> float:
    return float(cost_before - cost_after)


def diagnostic_rate(diagnostics: Sequence[Sequence[str]]) -> float:
    if not diagnostics:
        return 0.0
    return float(sum(bool(d) for d in diagnostics) / len(diagnostics))


def mean_interpretability(indices: Sequence[float]) -> float | None:
    if not indices:
        return None
    return float(np.mean(indices))
