"""Power-aware eligibility for rule-effect experiments."""

from __future__ import annotations

import math
from dataclasses import dataclass
from statistics import NormalDist


@dataclass(frozen=True)
class PowerAnalysis:
    effect: float
    support: float
    alpha: float
    target_power: float
    required_objects: int
    expected_active_objects: int
    achieved_power: float
    eligible: bool


def binary_effect_power(
    *,
    effect: float,
    support: float,
    total_objects: int,
    alpha: float = 0.05,
    target_power: float = 0.80,
    baseline_rate: float = 0.5,
) -> PowerAnalysis:
    if not 0.0 < support <= 1.0 or total_objects <= 0 or not 0.0 < alpha < 1.0 or not 0.0 < target_power < 1.0:
        raise ValueError("invalid power-analysis inputs")
    absolute = abs(effect)
    if absolute <= 0.0:
        return PowerAnalysis(effect, support, alpha, target_power, 2**31 - 1, int(total_objects * support), 0.0, False)
    variance = baseline_rate * (1.0 - baseline_rate) + min(0.999, max(0.001, baseline_rate + absolute)) * (1.0 - min(0.999, max(0.001, baseline_rate + absolute)))
    z_alpha = NormalDist().inv_cdf(1.0 - alpha / 2.0)
    z_power = NormalDist().inv_cdf(target_power)
    active_required = math.ceil(variance * (z_alpha + z_power) ** 2 / (absolute**2))
    required_total = math.ceil(active_required / support)
    active_observed = max(1, int(total_objects * support))
    noncentral = absolute * math.sqrt(active_observed / max(variance, 1e-12))
    achieved = float(NormalDist().cdf(noncentral - z_alpha))
    return PowerAnalysis(effect, support, alpha, target_power, required_total, active_observed, achieved, achieved >= target_power)
