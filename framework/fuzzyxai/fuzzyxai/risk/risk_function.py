from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Sequence

import numpy as np

DEFAULT_RISK_WEIGHTS: dict[str, float] = {
    'predicted_risk': 0.30,
    'uncertainty': 0.25,
    'reduction_loss': 0.15,
    'interpretability_gap': 0.20,
    'diagnostic': 0.10,
}


def clip01(value: float) -> float:
    return max(0.0, min(1.0, float(value)))


def normalize_risk_weights(weights: Mapping[str, float] | None = None) -> dict[str, float]:
    """Return non-negative risk weights normalized to the simplex."""
    merged = dict(DEFAULT_RISK_WEIGHTS)
    if weights:
        merged.update({str(k): max(0.0, float(v)) for k, v in weights.items()})
    total = sum(merged.values())
    if total <= 0.0:
        return dict(DEFAULT_RISK_WEIGHTS)
    return {key: value / total for key, value in merged.items()}


@dataclass(frozen=True)
class ApplicationRiskBreakdown:
    predicted_risk: float
    uncertainty: float
    pre_interpretability: float
    interpretability_gap: float
    reduction_loss: float
    diagnostic: float
    weights: dict[str, float]
    rho: float

    def as_dict(self) -> dict[str, float | dict[str, float]]:
        return {
            'predicted_risk': self.predicted_risk,
            'uncertainty': self.uncertainty,
            'pre_interpretability': self.pre_interpretability,
            'interpretability_gap': self.interpretability_gap,
            'reduction_loss': self.reduction_loss,
            'diagnostic': self.diagnostic,
            'weights': dict(self.weights),
            'rho': self.rho,
        }


def compute_application_risk(
    predicted_risk: float,
    uncertainty: float,
    pre_interpretability: float | None = None,
    reduction_loss: float = 0.0,
    diagnostics: Sequence[str] | None = None,
    weights: Mapping[str, float] | None = None,
    *,
    interpretability: float | None = None,
) -> ApplicationRiskBreakdown:
    """Compute rho(x) from the pre-risk explanation state.

    `pre_interpretability` is I_pre, not I_final. The final composition index is
    reported after the action explanation is built and must not feed this score.
    The `interpretability` keyword is kept as a backwards-compatible alias.
    """
    if pre_interpretability is None:
        if interpretability is None:
            raise TypeError('pre_interpretability is required')
        pre_interpretability = interpretability
    diagnostics = list(diagnostics or [])
    w = normalize_risk_weights(weights)
    pre_i = clip01(pre_interpretability)
    components = {
        'predicted_risk': clip01(predicted_risk),
        'uncertainty': clip01(uncertainty),
        'interpretability_gap': clip01(1.0 - pre_i),
        'reduction_loss': clip01(reduction_loss),
        'diagnostic': 1.0 if diagnostics else 0.0,
    }
    rho = clip01(sum(w[key] * components[key] for key in components))
    return ApplicationRiskBreakdown(weights=w, rho=rho, pre_interpretability=pre_i, **components)


def expected_action_costs(proba: Sequence[float], cost_matrix: Mapping[str, Sequence[float]]) -> dict[str, float]:
    """Compute E[C(a,y)|x] for each action from class probabilities."""
    p = np.asarray(proba, dtype=float).reshape(-1)
    if p.size == 0:
        raise ValueError('proba must contain at least one class probability')
    total = float(p.sum())
    if total <= 0.0:
        raise ValueError('proba must have positive sum')
    p = p / total
    costs: dict[str, float] = {}
    for action, row in cost_matrix.items():
        c = np.asarray(row, dtype=float).reshape(-1)
        if c.size != p.size:
            raise ValueError(f'cost row for {action!r} has length {c.size}, expected {p.size}')
        costs[str(action)] = float(np.dot(p, c))
    if not costs:
        raise ValueError('cost_matrix must contain at least one action')
    return costs


def choose_min_expected_cost(proba: Sequence[float], cost_matrix: Mapping[str, Sequence[float]]) -> tuple[str, dict[str, float]]:
    costs = expected_action_costs(proba, cost_matrix)
    action = min(costs, key=costs.get)
    return action, costs
