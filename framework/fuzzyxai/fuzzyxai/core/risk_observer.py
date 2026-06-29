from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping


@dataclass(frozen=True)
class RiskResult:
    rho: float
    action: str
    chi_r_crit: int


def compute_risk(components: Mapping[str, float], weights: Mapping[str, float]) -> float:
    missing = set(components) - set(weights)
    if missing:
        raise ValueError(f"Missing risk weights for components: {sorted(missing)}")
    total_weight = sum(float(weights[key]) for key in components)
    if total_weight <= 0:
        raise ValueError("Risk weights must have positive sum")
    value = sum(float(components[key]) * float(weights[key]) for key in components) / total_weight
    return round(value, 6)


def compute_action(rho: float, chi_r_crit: int, thresholds: Mapping[str, float]) -> str:
    if int(chi_r_crit) == 1:
        return "block"
    theta_1 = float(thresholds["theta_1"])
    theta_2 = float(thresholds["theta_2"])
    theta_3 = float(thresholds["theta_3"])
    theta_4 = float(thresholds["theta_4"])
    if not 0 <= theta_1 < theta_2 < theta_3 < theta_4 <= 1:
        raise ValueError("Expected 0 <= theta_1 < theta_2 < theta_3 < theta_4 <= 1")
    rho = float(rho)
    if rho < theta_1:
        return "accept"
    if rho < theta_2:
        return "lower_confidence"
    if rho < theta_3:
        return "request_more_data"
    if rho < theta_4:
        return "defer_to_human"
    return "block"


def observe_risk(components: Mapping[str, float], weights: Mapping[str, float], thresholds: Mapping[str, float], chi_r_crit: int) -> RiskResult:
    rho = compute_risk(components, weights)
    return RiskResult(rho=rho, action=compute_action(rho, chi_r_crit, thresholds), chi_r_crit=int(chi_r_crit))
