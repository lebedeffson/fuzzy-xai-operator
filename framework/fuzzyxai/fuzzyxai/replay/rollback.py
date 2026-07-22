"""Fail-closed rollback decisions for replay and canary operation."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class RollbackThresholds:
    false_block_ceiling: float = 0.01
    review_rate_ceiling: float = 0.30
    route_fault_rate_ceiling: float = 0.20
    calibration_deterioration_ceiling: float = 0.10


@dataclass(frozen=True)
class RollbackDecision:
    rollback: bool
    reason_codes: tuple[str, ...]


def evaluate_rollback(
    *,
    false_block_rate: float,
    review_rate: float,
    route_fault_rate: float,
    calibration_deterioration: float,
    thresholds: RollbackThresholds = RollbackThresholds(),
) -> RollbackDecision:
    reasons = []
    if false_block_rate > thresholds.false_block_ceiling:
        reasons.append("FALSE_BLOCK_CEILING_EXCEEDED")
    if review_rate > thresholds.review_rate_ceiling:
        reasons.append("REVIEW_RATE_CEILING_EXCEEDED")
    if route_fault_rate > thresholds.route_fault_rate_ceiling:
        reasons.append("ROUTE_FAULT_SPIKE")
    if calibration_deterioration > thresholds.calibration_deterioration_ceiling:
        reasons.append("CALIBRATION_DETERIORATION")
    return RollbackDecision(bool(reasons), tuple(reasons))
