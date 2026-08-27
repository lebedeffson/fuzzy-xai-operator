from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass


@dataclass(frozen=True)
class RiskResult:
    rho: float
    action: str
    chi_r_crit: int
    candidate_action: str = ""
    action_reason: str = ""


CANONICAL_RISK_COMPONENTS = (
    "rho_p",
    "u_M",
    "one_minus_I_pre",
    "Delta",
    "chi_R",
)


def compute_risk(components: Mapping[str, float], weights: Mapping[str, float]) -> float:
    """Legacy normalized compatibility score; not canonical P19 rho.

    Retained for sealed pre-P19 scenario engines. New public code must use
    :func:`compute_canonical_risk`.
    """
    missing = set(components) - set(weights)
    if missing:
        raise ValueError(f"Missing risk weights for components: {sorted(missing)}")
    total_weight = sum(float(weights[key]) for key in components)
    if total_weight <= 0:
        raise ValueError("Risk weights must have positive sum")
    value = sum(float(components[key]) * float(weights[key]) for key in components) / total_weight
    return round(value, 6)


def compute_canonical_risk(components: Mapping[str, float], weights: Mapping[str, float]) -> float:
    """Compute the strict five-component dissertation rho without renormalization."""

    missing_components = set(CANONICAL_RISK_COMPONENTS) - set(components)
    extra_components = set(components) - set(CANONICAL_RISK_COMPONENTS)
    if missing_components or extra_components:
        raise ValueError(
            "canonical rho requires exactly "
            f"{list(CANONICAL_RISK_COMPONENTS)}; missing={sorted(missing_components)}, "
            f"extra={sorted(extra_components)}"
        )
    if set(weights) != set(CANONICAL_RISK_COMPONENTS):
        raise ValueError("canonical rho weights must match the five canonical components")
    if any(float(value) < 0 for value in weights.values()):
        raise ValueError("canonical rho weights must be non-negative")
    if abs(sum(float(value) for value in weights.values()) - 1.0) > 1e-9:
        raise ValueError("canonical rho weights must sum to one; implicit renormalization is forbidden")
    return sum(float(weights[key]) * max(0.0, min(1.0, float(components[key]))) for key in CANONICAL_RISK_COMPONENTS)


def _candidate_action(rho: float, thresholds: Mapping[str, float], action_policy: Mapping[str, str] | None = None) -> tuple[str, str]:
    theta_1 = float(thresholds["theta_1"])
    theta_2 = float(thresholds["theta_2"])
    theta_3 = float(thresholds["theta_3"])
    theta_4 = float(thresholds["theta_4"])
    if not 0 <= theta_1 < theta_2 < theta_3 < theta_4 <= 1:
        raise ValueError("Expected 0 <= theta_1 < theta_2 < theta_3 < theta_4 <= 1")
    policy = dict(action_policy or {})
    middle = policy.get("theta_2_to_theta_3", "request_more_data")
    upper = policy.get("theta_3_to_theta_4", "defer_to_human")
    rho = float(rho)
    if rho < theta_1:
        return "accept", "rho is below theta_1"
    if rho < theta_2:
        return "lower_confidence", "rho is between theta_1 and theta_2"
    if rho < theta_3:
        return middle, f"rho is between theta_2 and theta_3; policy selected {middle}"
    if rho < theta_4:
        return upper, f"rho is between theta_3 and theta_4; policy selected {upper}"
    return "block", "rho is at or above theta_4"


def compute_action(rho: float, chi_r_crit: int, thresholds: Mapping[str, float], action_policy: Mapping[str, str] | None = None) -> str:
    candidate, _ = _candidate_action(rho, thresholds, action_policy)
    return "block" if int(chi_r_crit) == 1 else candidate


def observe_risk(components: Mapping[str, float], weights: Mapping[str, float], thresholds: Mapping[str, float], chi_r_crit: int, action_policy: Mapping[str, str] | None = None) -> RiskResult:
    """Public canonical P19 risk observer."""

    rho = compute_canonical_risk(components, weights)
    candidate, reason = _candidate_action(rho, thresholds, action_policy)
    critical = int(chi_r_crit)
    return RiskResult(rho=rho, action="block" if critical == 1 else candidate, chi_r_crit=critical, candidate_action=candidate, action_reason="critical structural override has priority" if critical == 1 else reason)


def observe_legacy_normalized_risk(components: Mapping[str, float], weights: Mapping[str, float], thresholds: Mapping[str, float], chi_r_crit: int) -> RiskResult:
    """Deprecated internal compatibility observer; result is not P19 rho."""

    score = compute_risk(components, weights)
    legacy_thresholds = dict(thresholds)
    legacy_thresholds.setdefault("theta_4", legacy_thresholds.get("theta_3", 1.0))
    candidate, reason = _candidate_action(score, legacy_thresholds)
    critical = int(chi_r_crit)
    return RiskResult(rho=score, action="block" if critical == 1 else candidate, chi_r_crit=critical, candidate_action=candidate, action_reason="critical structural override has priority" if critical == 1 else reason)
