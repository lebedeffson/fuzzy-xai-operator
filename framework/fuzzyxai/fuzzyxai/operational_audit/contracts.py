from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class AuditAction(str, Enum):
    BLOCK = "block"
    REPAIR_THEN_RETRY = "repair_then_retry"
    REVIEW = "review"
    ACCEPT = "accept"


class RouteOutcome(str, Enum):
    VALID = "valid_route"
    KNOWN_FAULT = "known_fault_type"
    UNKNOWN_FAULT = "unknown_structural_fault"
    INSUFFICIENT = "insufficient_evidence"


@dataclass(frozen=True)
class RouteArtifact:
    artifact_id: str
    model_id: str
    explainer_model_id: str
    calibration_model_id: str
    preprocessing_steps: tuple[str, ...]
    feature_schema: tuple[str, ...]
    explainer_feature_schema: tuple[str, ...]
    canonical_sha256: str
    observed_sha256: str
    reduction_source_id: str
    reduction_target_source_id: str
    reference_population_id: str
    expected_reference_population_id: str
    provenance_nodes: tuple[str, ...]
    mandatory_provenance_nodes: tuple[str, ...]
    dictionary_version: str
    expected_dictionary_version: str
    evidence_complete: bool = True


@dataclass(frozen=True)
class RouteAssessment:
    outcome: RouteOutcome
    irreparable_fault: bool
    repairable_fault: bool
    family: str | None
    confidence: float
    damaged_regions: tuple[str, ...]
    violations: tuple[str, ...]
    reason_codes: tuple[str, ...]


@dataclass(frozen=True)
class PredictiveState:
    risk: float
    requires_review: bool
    threshold: float


@dataclass(frozen=True)
class RepairPlan:
    candidate_actions: tuple[str, ...]
    repairable: bool
    requires_recertification: bool = True


@dataclass(frozen=True)
class OperationalDecision:
    action: AuditAction
    reason_codes: tuple[str, ...]
    route: RouteAssessment
    predictive: PredictiveState
    repair_plan: RepairPlan
    audit_trace: bytes
