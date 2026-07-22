"""Fail-closed H6 formative gate."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class H6FormativeGate:
    detection_rate: float
    sign_accuracy: float
    false_discovery_rate: float
    power_eligible_fraction: float
    passed: bool
    confirmatory_opening_allowed: bool


def evaluate_h6_formative_gate(
    *,
    detection_rate: float,
    sign_accuracy: float,
    false_discovery_rate: float,
    power_eligible_fraction: float,
) -> H6FormativeGate:
    values = (detection_rate, sign_accuracy, false_discovery_rate, power_eligible_fraction)
    if any(not 0.0 <= value <= 1.0 for value in values):
        raise ValueError("H6 gate metrics must be in [0, 1]")
    passed = detection_rate >= 0.80 and sign_accuracy >= 0.90 and false_discovery_rate <= 0.10
    return H6FormativeGate(detection_rate, sign_accuracy, false_discovery_rate, power_eligible_fraction, passed, passed)
