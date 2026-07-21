"""Typed contracts shared by the full empirical-validation protocols."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import Literal, Mapping, Sequence


EvidenceStatus = Literal["measured", "controlled", "planned_not_run", "not_available", "failed"]
GateStatus = Literal["PASS", "BLOCKED", "FAIL"]
Action = Literal["accept", "review", "block"]


class CriticalRuptureType(str, Enum):
    MISSING_REQUIRED_EVIDENCE = "missing_required_evidence"
    FORBIDDEN_RULE_CONFLICT = "forbidden_rule_conflict"
    UNVERIFIED_PROVENANCE = "unverified_provenance"
    REPRESENTATION_UNDERCOVERAGE = "representation_undercoverage"
    REDUCTION_LOSS_EXCEEDED = "reduction_loss_exceeded"
    DISTRIBUTION_SHIFT = "distribution_shift"
    UNSTABLE_EXPLANATION = "unstable_explanation"
    CROSS_MODEL_DISAGREEMENT = "cross_model_disagreement"


@dataclass(frozen=True)
class DatasetSpec:
    dataset_id: str
    modality: Literal["tabular", "image", "text", "time_series"]
    task: str
    minimum_objects: int
    source: str
    license: str
    version: str
    acquisition_date: str
    generator: str | None = None
    expected_sha256: str | None = None
    limitations: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True)
class CalibrationConfig:
    accept_threshold: float
    review_threshold: float
    block_threshold: float
    conflict_weight: float
    incompleteness_weight: float
    shift_weight: float
    reduction_loss_weight: float
    coverage_threshold: float
    stability_threshold: float
    critical_rupture_threshold: float
    critical_error_cost: float
    wrong_auto_cost: float
    review_cost: float
    false_block_cost: float

    def __post_init__(self) -> None:
        probability_fields = (
            self.accept_threshold,
            self.review_threshold,
            self.block_threshold,
            self.coverage_threshold,
            self.stability_threshold,
            self.critical_rupture_threshold,
        )
        if any(value < 0.0 or value > 1.0 for value in probability_fields):
            raise ValueError("probability thresholds must be within [0, 1]")
        if self.accept_threshold < self.review_threshold:
            raise ValueError("accept_threshold must not be below review_threshold")
        costs = (
            self.critical_error_cost,
            self.wrong_auto_cost,
            self.review_cost,
            self.false_block_cost,
        )
        if any(value < 0.0 for value in costs):
            raise ValueError("decision costs must be non-negative")

    def to_dict(self) -> dict[str, float]:
        return asdict(self)


@dataclass(frozen=True)
class CriticalRupture:
    rupture_type: CriticalRuptureType
    object_id: str
    evidence_refs: tuple[str, ...]
    measured_value: float | None = None
    threshold: float | None = None
    detail: str = ""

    def __post_init__(self) -> None:
        if not self.evidence_refs:
            raise ValueError("a critical rupture requires evidence_refs")

    def to_dict(self) -> dict[str, object]:
        payload = asdict(self)
        payload["rupture_type"] = self.rupture_type.value
        return payload


@dataclass(frozen=True)
class ExperimentGate:
    experiment_id: str
    status: GateStatus
    evidence_status: EvidenceStatus
    checks: Mapping[str, bool]
    evidence_files: tuple[str, ...] = ()
    limitations: tuple[str, ...] = ()
    metrics: Mapping[str, float | int | str | bool | None] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.status == "PASS" and self.evidence_status not in {"measured", "controlled"}:
            raise ValueError("PASS requires measured or controlled evidence")
        if self.status == "PASS" and not all(self.checks.values()):
            raise ValueError("PASS requires every declared check to pass")

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True)
class PolicyOutcome:
    policy_id: str
    action: Action
    predicted_label: int | str | None
    true_label: int | str | None
    is_critical: bool
    explanation_time_seconds: float
    reason: str


@dataclass(frozen=True)
class ExperimentRunManifest:
    schema_version: str
    profile: Literal["smoke", "full"]
    commit: str
    branch: str
    seed: int
    threads: int
    experiments: Sequence[ExperimentGate]
    external_gates: Mapping[str, str]

    @property
    def release_status(self) -> GateStatus:
        if any(item.status == "FAIL" for item in self.experiments):
            return "FAIL"
        if any(item.status != "PASS" for item in self.experiments):
            return "BLOCKED"
        if any(status not in {"PASS", "completed"} for status in self.external_gates.values()):
            return "BLOCKED"
        return "PASS"

    def to_dict(self) -> dict[str, object]:
        payload = asdict(self)
        payload["release_status"] = self.release_status
        payload["tag_allowed"] = self.release_status == "PASS"
        return payload
