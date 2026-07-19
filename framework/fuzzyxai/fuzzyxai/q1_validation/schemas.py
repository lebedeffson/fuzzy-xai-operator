"""Typed contracts for the independent Q1 validation cycle."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import Mapping, Sequence


class PartitionRole(str, Enum):
    TRAIN = "train"
    VALIDATION = "validation"
    TEST = "test"


class OperationKind(str, Enum):
    FIT = "fit"
    SELECT = "select"
    CALIBRATE = "calibrate"
    EVALUATE = "evaluate"


class ClaimStatus(str, Enum):
    SUPPORTED = "supported"
    NOT_SUPPORTED = "not_supported"
    INCONCLUSIVE = "inconclusive"
    EXTERNAL_GATE = "external_gate"


class EvidenceOrigin(str, Enum):
    MEASURED_REAL = "measured_real"
    MEASURED_CONTROLLED = "measured_controlled"
    DERIVED = "derived"
    EXTERNAL = "external"


@dataclass(frozen=True)
class SplitUseRecord:
    operation_id: str
    operation: OperationKind
    partitions: tuple[PartitionRole, ...]
    input_hashes: Mapping[str, str]

    def __post_init__(self) -> None:
        if not self.operation_id:
            raise ValueError("operation_id is required")
        if not self.partitions:
            raise ValueError("at least one partition is required")
        if self.operation in {OperationKind.FIT, OperationKind.SELECT, OperationKind.CALIBRATE} and PartitionRole.TEST in self.partitions:
            raise ValueError(f"test partition cannot be used for {self.operation.value}")
        missing = [role.value for role in self.partitions if role.value not in self.input_hashes]
        if missing:
            raise ValueError(f"missing partition hashes: {missing}")

    def to_dict(self) -> dict[str, object]:
        return {
            "operation_id": self.operation_id,
            "operation": self.operation.value,
            "partitions": [item.value for item in self.partitions],
            "input_hashes": dict(self.input_hashes),
        }


@dataclass(frozen=True)
class Q1CalibrationConfig:
    accept_threshold: float
    review_threshold: float
    conflict_weight: float
    missing_weight: float
    shift_weight: float
    instability_weight: float
    reduction_loss_weight: float
    critical_rupture_threshold: float
    escalation_threshold: float
    risk_cost_critical: float
    risk_cost_wrong_auto: float
    risk_cost_review: float
    risk_cost_false_block: float

    def __post_init__(self) -> None:
        thresholds = (
            self.accept_threshold,
            self.review_threshold,
            self.critical_rupture_threshold,
            self.escalation_threshold,
        )
        weights = (
            self.conflict_weight,
            self.missing_weight,
            self.shift_weight,
            self.instability_weight,
            self.reduction_loss_weight,
        )
        costs = (
            self.risk_cost_critical,
            self.risk_cost_wrong_auto,
            self.risk_cost_review,
            self.risk_cost_false_block,
        )
        if any(not 0.0 <= value <= 1.0 for value in thresholds):
            raise ValueError("thresholds must be within [0, 1]")
        if self.accept_threshold < self.review_threshold:
            raise ValueError("accept_threshold must be at least review_threshold")
        if any(value < 0.0 for value in (*weights, *costs)):
            raise ValueError("weights and costs must be non-negative")

    def to_dict(self) -> dict[str, float]:
        return asdict(self)


@dataclass(frozen=True)
class FidelityPair:
    object_id: str
    explainer: str
    baseline: float
    wrapped: float
    same_model: bool
    same_background: bool
    same_budget: bool
    same_seed: bool

    def __post_init__(self) -> None:
        if not all((self.same_model, self.same_background, self.same_budget, self.same_seed)):
            raise ValueError("a fidelity comparison must be strictly paired")

    @property
    def difference(self) -> float:
        return self.wrapped - self.baseline


@dataclass(frozen=True)
class HypothesisResult:
    hypothesis_id: str
    status: ClaimStatus
    metrics: Mapping[str, float | int | str | bool | None]
    evidence_files: tuple[str, ...]
    limitations: tuple[str, ...]
    allowed_wording: str
    forbidden_wording: str
    origin: EvidenceOrigin

    def __post_init__(self) -> None:
        if not self.hypothesis_id or not self.evidence_files:
            raise ValueError("hypothesis result requires an id and evidence files")
        if self.status is ClaimStatus.SUPPORTED and self.origin is EvidenceOrigin.EXTERNAL:
            raise ValueError("external placeholders cannot support a measured claim")

    def to_dict(self) -> dict[str, object]:
        payload = asdict(self)
        payload["status"] = self.status.value
        payload["origin"] = self.origin.value
        return payload


@dataclass(frozen=True)
class ExternalGate:
    gate_id: str
    status: str
    participant_count: int = 0
    evidence_files: tuple[str, ...] = field(default_factory=tuple)

    def __post_init__(self) -> None:
        valid = {"planned_not_run", "pending_external_review", "completed", "failed"}
        if self.status not in valid:
            raise ValueError(f"invalid external gate status: {self.status}")
        if self.status == "completed" and (self.participant_count <= 0 or not self.evidence_files):
            raise ValueError("a completed external gate requires participant records and evidence")


def ensure_unique_ids(values: Sequence[str], *, label: str) -> None:
    if len(set(values)) != len(values):
        raise ValueError(f"duplicate {label} identifiers")
