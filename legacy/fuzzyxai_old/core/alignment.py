from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping


@dataclass(frozen=True)
class AlignmentResult:
    gamma: float
    gamma_max: float
    delta_t: float
    certified: bool


def compute_gamma(components: Mapping[str, float], weights: Mapping[str, float]) -> float:
    """Weighted component disagreement for T_ij alignment."""
    missing = set(components) - set(weights)
    if missing:
        raise ValueError(f"Missing weights for components: {sorted(missing)}")
    total_weight = sum(float(weights[key]) for key in components)
    if total_weight <= 0:
        raise ValueError("Alignment weights must have positive sum")
    value = sum(float(components[key]) * float(weights[key]) for key in components) / total_weight
    return round(value, 6)


def compute_alignment(
    components: Mapping[str, float],
    weights: Mapping[str, float],
    gamma_max: float,
    delta_t: float,
    delta_max: float,
) -> AlignmentResult:
    gamma = compute_gamma(components, weights)
    return AlignmentResult(
        gamma=gamma,
        gamma_max=float(gamma_max),
        delta_t=float(delta_t),
        certified=gamma <= float(gamma_max) and float(delta_t) <= float(delta_max),
    )


def compute_gamma_route(p: float, alpha_mean: float, feature_support: float, mode: str = "model_to_explanations") -> float:
    """Route disagreement used by the GIS INTEGRO control example."""
    if mode != "model_to_explanations":
        raise ValueError(f"Unsupported route mode: {mode}")
    gamma = abs(float(p) - float(feature_support))
    return round(gamma, 6)
