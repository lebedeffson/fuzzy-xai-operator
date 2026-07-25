from __future__ import annotations

from dataclasses import dataclass
from itertools import product

from .baselines import RepairAction, select_global_minimum_cut
from .cost_models import (
    DEFAULT_COST_WEIGHTS,
    CostCalibration,
    CostWeights,
    action_cost,
)

ALPHAS = (0.0, 0.25, 0.5, 1.0)
BETAS = (0.0, 0.25, 0.5, 1.0)
GAMMAS = (0.0, 0.5, 1.0)


@dataclass(frozen=True)
class CostSensitivityRow:
    alpha: float
    beta: float
    gamma: float
    selected_cut: tuple[str, ...]
    selected_cut_cost: float
    alternative_optimal_cuts: tuple[tuple[str, ...], ...]
    selection_changed: bool
    repair_success: bool
    cost_regret: float


def run_cost_sensitivity(
    actions: tuple[RepairAction, ...],
    obligations: frozenset[str],
    calibration: CostCalibration,
    *,
    nominal_weights: CostWeights = DEFAULT_COST_WEIGHTS,
) -> tuple[CostSensitivityRow, ...]:
    nominal_cost = lambda action: action_cost(
        action, "hybrid", calibration, nominal_weights
    )
    nominal = select_global_minimum_cut(actions, obligations, nominal_cost)
    rows = []
    for alpha, beta, gamma in product(ALPHAS, BETAS, GAMMAS):
        weights = CostWeights(alpha, beta, gamma)
        cost = lambda action, w=weights: action_cost(action, "hybrid", calibration, w)
        selected = select_global_minimum_cut(actions, obligations, cost)
        action_by_id = {action.action_id: action for action in actions}
        nominal_under_scenario = sum(cost(action_by_id[item]) for item in nominal.action_ids)
        rows.append(
            CostSensitivityRow(
                alpha=alpha,
                beta=beta,
                gamma=gamma,
                selected_cut=selected.action_ids,
                selected_cut_cost=selected.predicted_cost,
                alternative_optimal_cuts=selected.equivalent_optimal_plans,
                selection_changed=(
                    selected.action_ids not in nominal.equivalent_optimal_plans
                ),
                repair_success=selected.feasible,
                cost_regret=max(0.0, nominal_under_scenario - selected.predicted_cost),
            )
        )
    return tuple(rows)
