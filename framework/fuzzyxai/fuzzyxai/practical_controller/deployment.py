"""Shadow, canary, delayed-label monitoring and deterministic rollback."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from typing import Mapping

from fuzzyxai.selective_observer import SelectiveAction

from .contracts import ActionAssessment


@dataclass(frozen=True)
class CanaryPolicy:
    policy_version: str
    traffic_fraction: float
    false_block_ceiling: float
    latency_ceiling_ms: float
    review_rate_ceiling: float
    route_fault_rate_ceiling: float
    calibration_deterioration_ceiling: float

    def __post_init__(self) -> None:
        normalized = (
            self.traffic_fraction,
            self.false_block_ceiling,
            self.review_rate_ceiling,
            self.route_fault_rate_ceiling,
            self.calibration_deterioration_ceiling,
        )
        if any(not 0.0 <= value <= 1.0 for value in normalized) or self.latency_ceiling_ms <= 0.0:
            raise ValueError("canary thresholds are invalid")


@dataclass(frozen=True)
class DeploymentRecord:
    object_id: str
    assessment: ActionAssessment
    actual_action: SelectiveAction
    latency_ms: float
    calibration_residual: float
    route_fault_observed: bool
    delayed_invalid_outcome: bool | None = None
    false_block: bool | None = None


@dataclass(frozen=True)
class MonitoringSnapshot:
    n_records: int
    labeled_records: int
    unsafe_accepts: int
    unnecessary_reviews: int
    false_blocks: int
    policy_regret: float
    calibration_drift: float
    action_changes: int
    review_rate: float
    route_fault_rate: float
    p95_latency_ms: float
    rollback_required: bool
    rollback_reason_codes: tuple[str, ...]


@dataclass
class ShadowCanaryMonitor:
    canary_policy: CanaryPolicy
    records: dict[str, DeploymentRecord] = field(default_factory=dict)
    previous_calibration_residual: float = 0.0
    active_policy_version: str | None = None
    rollback_policy_version: str | None = None

    def in_canary(self, object_id: str) -> bool:
        bucket = int(hashlib.sha256(object_id.encode()).hexdigest()[:12], 16) / float(16**12 - 1)
        return bucket < self.canary_policy.traffic_fraction

    def record_shadow(
        self,
        object_id: str,
        assessment: ActionAssessment,
        *,
        actual_action: SelectiveAction,
        latency_ms: float,
        calibration_residual: float,
        route_fault_observed: bool,
    ) -> None:
        if object_id in self.records:
            raise ValueError("shadow record is idempotent and cannot be overwritten")
        self.records[object_id] = DeploymentRecord(
            object_id=object_id,
            assessment=assessment,
            actual_action=actual_action,
            latency_ms=latency_ms,
            calibration_residual=calibration_residual,
            route_fault_observed=route_fault_observed,
        )

    def attach_delayed_label(self, object_id: str, *, invalid_outcome: bool, false_block: bool) -> None:
        record = self.records[object_id]
        if record.delayed_invalid_outcome is not None:
            raise ValueError("delayed labels cannot be rewritten")
        self.records[object_id] = DeploymentRecord(
            **{
                **record.__dict__,
                "delayed_invalid_outcome": invalid_outcome,
                "false_block": false_block,
            }
        )

    def snapshot(self) -> MonitoringSnapshot:
        if not self.records:
            raise ValueError("monitoring snapshot requires records")
        records = list(self.records.values())
        labeled = [record for record in records if record.delayed_invalid_outcome is not None]
        unsafe = sum(record.assessment.action is SelectiveAction.ACCEPT and bool(record.delayed_invalid_outcome) for record in labeled)
        unnecessary = sum(
            record.assessment.action in {SelectiveAction.SHORT_REVIEW, SelectiveAction.FULL_REVIEW}
            and not bool(record.delayed_invalid_outcome)
            for record in labeled
        )
        false_blocks = sum(bool(record.false_block) for record in labeled)
        action_changes = sum(record.assessment.action is not record.actual_action for record in records)
        review_rate = sum(
            record.assessment.action in {SelectiveAction.SHORT_REVIEW, SelectiveAction.FULL_REVIEW} for record in records
        ) / len(records)
        route_fault_rate = sum(record.route_fault_observed for record in records) / len(records)
        calibration = sum(record.calibration_residual for record in records) / len(records)
        p95 = _quantile([record.latency_ms for record in records], 0.95)
        false_block_rate = false_blocks / max(1, len(labeled))
        reasons = []
        if false_block_rate > self.canary_policy.false_block_ceiling:
            reasons.append("FALSE_BLOCK_CEILING_EXCEEDED")
        if p95 > self.canary_policy.latency_ceiling_ms:
            reasons.append("LATENCY_CEILING_EXCEEDED")
        if review_rate > self.canary_policy.review_rate_ceiling:
            reasons.append("REVIEW_RATE_CEILING_EXCEEDED")
        if route_fault_rate > self.canary_policy.route_fault_rate_ceiling:
            reasons.append("ROUTE_FAULT_SPIKE")
        if calibration - self.previous_calibration_residual > self.canary_policy.calibration_deterioration_ceiling:
            reasons.append("CALIBRATION_DETERIORATION")
        regret = (unsafe + 0.25 * unnecessary + false_blocks) / max(1, len(labeled))
        return MonitoringSnapshot(
            n_records=len(records),
            labeled_records=len(labeled),
            unsafe_accepts=unsafe,
            unnecessary_reviews=unnecessary,
            false_blocks=false_blocks,
            policy_regret=regret,
            calibration_drift=calibration - self.previous_calibration_residual,
            action_changes=action_changes,
            review_rate=review_rate,
            route_fault_rate=route_fault_rate,
            p95_latency_ms=p95,
            rollback_required=bool(reasons),
            rollback_reason_codes=tuple(reasons),
        )

    def rollback(self, policies: Mapping[str, object]) -> str:
        snapshot = self.snapshot()
        if not snapshot.rollback_required:
            raise ValueError("rollback is not permitted without a breached canary ceiling")
        if not self.rollback_policy_version or self.rollback_policy_version not in policies:
            raise ValueError("rollback policy is unavailable")
        self.active_policy_version = self.rollback_policy_version
        return self.active_policy_version


def _quantile(values: list[float], quantile: float) -> float:
    ordered = sorted(values)
    index = min(len(ordered) - 1, max(0, round((len(ordered) - 1) * quantile)))
    return float(ordered[index])
