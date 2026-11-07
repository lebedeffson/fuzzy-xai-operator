from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Mapping, Sequence

import numpy as np

DEFAULT_RISK_WEIGHTS: dict[str, float] = {
    'predicted_risk': 0.30,
    'uncertainty': 0.25,
    'reduction_loss': 0.15,
    'interpretability_gap': 0.20,
    'diagnostic': 0.10,
}


def _finite_risk_number(value: float, name: str) -> float:
    if isinstance(value, bool):
        raise ValueError(f"{name} must be a finite numeric value")
    try:
        parsed = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"{name} must be a finite numeric value") from exc
    if not math.isfinite(parsed):
        raise ValueError(f"{name} must be finite")
    return parsed


def clip01(value: float) -> float:
    checked = _finite_risk_number(value, "risk component")
    return max(0.0, min(1.0, checked))


def normalize_risk_weights(weights: Mapping[str, float] | None = None) -> dict[str, float]:
    """Validate five weights; implicit renormalization is forbidden in P19."""
    merged = dict(DEFAULT_RISK_WEIGHTS)
    if weights:
        unknown = set(weights) - set(DEFAULT_RISK_WEIGHTS)
        if unknown:
            raise ValueError(f"unknown risk component weights: {sorted(unknown)}")
        merged.update({str(k): _finite_risk_number(v, f"weight:{k}") for k, v in weights.items()})
    if any(value < 0.0 for value in merged.values()):
        raise ValueError("risk weights must be non-negative")
    total = sum(merged.values())
    if abs(total - 1.0) > 1e-9:
        raise ValueError("risk weights must sum to one; implicit renormalization is forbidden")
    return merged


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
    if not np.all(np.isfinite(p)) or np.any(p < 0.0):
        raise ValueError('proba must contain finite non-negative probabilities')
    total = float(p.sum())
    if not math.isfinite(total) or total <= 0.0:
        raise ValueError('proba must have finite positive sum')
    p = p / total
    costs: dict[str, float] = {}
    for action, row in cost_matrix.items():
        c = np.asarray(row, dtype=float).reshape(-1)
        if c.size != p.size:
            raise ValueError(f'cost row for {action!r} has length {c.size}, expected {p.size}')
        if not np.all(np.isfinite(c)):
            raise ValueError(f'cost row for {action!r} contains non-finite values')
        costs[str(action)] = float(np.dot(p, c))
    if not costs:
        raise ValueError('cost_matrix must contain at least one action')
    return costs


def choose_min_expected_cost(proba: Sequence[float], cost_matrix: Mapping[str, Sequence[float]]) -> tuple[str, dict[str, float]]:
    costs = expected_action_costs(proba, cost_matrix)
    action = min(costs, key=costs.get)
    return action, costs
