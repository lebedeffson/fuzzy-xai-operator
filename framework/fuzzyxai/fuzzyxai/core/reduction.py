from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping


@dataclass(frozen=True)
class ReductionResult:
    delta: float
    delta_max: float
    allowed: bool


def compute_reduction_loss(components: Mapping[str, float], weights: Mapping[str, float]) -> float:
    missing = set(components) - set(weights)
    if missing:
        raise ValueError(f"Missing weights for reduction components: {sorted(missing)}")
    total_weight = sum(float(weights[key]) for key in components)
    if total_weight <= 0:
        raise ValueError("Reduction weights must have positive sum")
    value = sum(float(components[key]) * float(weights[key]) for key in components) / total_weight
    return round(value, 6)


def compute_reduction(components: Mapping[str, float], weights: Mapping[str, float], delta_max: float) -> ReductionResult:
    delta = compute_reduction_loss(components, weights)
    return ReductionResult(delta=delta, delta_max=float(delta_max), allowed=delta <= float(delta_max))
