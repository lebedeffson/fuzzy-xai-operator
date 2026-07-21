"""Typed production contracts for the budgeted practical controller."""

from __future__ import annotations

import re
from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import Mapping, Sequence

from fuzzyxai.selective_observer import SelectiveAction


SHA256_RE = re.compile(r"^[0-9a-f]{64}$")


class HardGuardStatus(str, Enum):
    CERTIFIED = "certified"
    REVIEW_REQUIRED = "review_required"
    BLOCKED = "blocked"


class DeploymentMode(str, Enum):
    ACTIVE = "active"
    SHADOW = "shadow"
    CANARY = "canary"


class CostProfileName(str, Enum):
    BALANCED = "balanced"
    UNSAFE_ACCEPT_SENSITIVE = "unsafe_accept_sensitive"
    REVIEW_EXPENSIVE = "review_expensive"


@dataclass(frozen=True)
class PredictionArtifact:
    object_id: str
    prediction: str
    confidence: float
    probabilities: tuple[float, ...]
    model_version: str
    calibration_residual: float = 0.0
    entropy: float = 0.0
    prediction_margin: float = 0.0
    boundary_distance: float = 1.0
    model_disagreement: float = 0.0
    shift_score: float = 0.0
    rare_group: bool = False

    def __post_init__(self) -> None:
        if not self.object_id or not self.prediction or not self.model_version:
            raise ValueError("prediction artifact identifiers are required")
        normalized = (
            self.confidence,
            self.calibration_residual,
            self.entropy,
            self.prediction_margin,
            self.boundary_distance,
            self.model_disagreement,
            self.shift_score,
        )
        if any(not 0.0 <= value <= 1.0 for value in normalized):
            raise ValueError("prediction risk channels must be in [0, 1]")
        if not self.probabilities or any(not 0.0 <= value <= 1.0 for value in self.probabilities):
            raise ValueError("probabilities must be a non-empty sequence in [0, 1]")


@dataclass(frozen=True)
class ExplanationArtifact:
    canonical_sha256: str
    explainer_version: str
    model_version: str
    explain_plan_version: str
    dictionary_version: str
    available_channels: tuple[str, ...]
    explainer_disagreement: float = 0.0
    seed_instability: float = 0.0
    bootstrap_instability: float = 0.0
    perturbation_instability: float = 0.0
    representation_loss: float = 0.0
    rule_redundancy: float = 0.0
    conflict_severity: float = 0.0

    def __post_init__(self) -> None:
        if not SHA256_RE.fullmatch(self.canonical_sha256):
            raise ValueError("canonical_sha256 must be a SHA256 digest")
        if not all((self.explainer_version, self.model_version, self.explain_plan_version, self.dictionary_version)):
            raise ValueError("explanation versions are required")
        channels = (
            self.explainer_disagreement,
            self.seed_instability,
            self.bootstrap_instability,
            self.perturbation_instability,
            self.representation_loss,
            self.rule_redundancy,
            self.conflict_severity,
        )
        if any(not 0.0 <= value <= 1.0 for value in channels):
            raise ValueError("explanation risk channels must be in [0, 1]")


@dataclass(frozen=True)
class RouteArtifacts:
    preprocessing_version: str
    calibration_version: str | None
    reference_population: str | None
    schema_version: str
    artifact_sha256: str
    observed_provenance_channels: tuple[str, ...]
    route_fault_type: str | None = None
    route_fault_source: str | None = None
    forbidden_rule_conflict: bool = False
    critical_data_quality_fault: bool = False
    natural_failure: str | None = None

    def __post_init__(self) -> None:
        if not self.preprocessing_version or not self.schema_version:
            raise ValueError("route versions are required")
        if not SHA256_RE.fullmatch(self.artifact_sha256):
            raise ValueError("artifact_sha256 must be a SHA256 digest")


@dataclass(frozen=True)
class DeploymentContext:
    expected_model_version: str
    expected_preprocessing_version: str
    expected_explainer_version: str
    expected_calibration_version: str
    expected_reference_population: str
    expected_schema_version: str
    expected_explain_plan_version: str
    expected_dictionary_version: str
    expected_artifact_sha256: str
    mandatory_provenance_channels: tuple[str, ...]
    maximum_reduction_loss: float
    policy_version: str
    mode: DeploymentMode = DeploymentMode.ACTIVE
    monitoring_hooks: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        versions = (
            self.expected_model_version,
            self.expected_preprocessing_version,
            self.expected_explainer_version,
            self.expected_calibration_version,
            self.expected_reference_population,
            self.expected_schema_version,
            self.expected_explain_plan_version,
            self.expected_dictionary_version,
            self.policy_version,
        )
        if not all(versions):
            raise ValueError("deployment contract versions are required")
        if not SHA256_RE.fullmatch(self.expected_artifact_sha256):
            raise ValueError("expected_artifact_sha256 must be a SHA256 digest")
        if not self.mandatory_provenance_channels:
            raise ValueError("at least one mandatory provenance channel is required")
        if not 0.0 <= self.maximum_reduction_loss <= 1.0:
            raise ValueError("maximum_reduction_loss must be in [0, 1]")


@dataclass(frozen=True)
class ReviewBudget:
    fraction: float
    current_review_fraction: float = 0.0

    def __post_init__(self) -> None:
        if self.fraction not in {0.05, 0.10, 0.20, 0.30, 1.0}:
            raise ValueError("review budget must be one of 5%, 10%, 20%, 30%, or unlimited")
        if not 0.0 <= self.current_review_fraction <= 1.0:
            raise ValueError("current review fraction must be in [0, 1]")


@dataclass(frozen=True)
class CostProfile:
    name: CostProfileName
    unsafe_accept: float
    short_review: float
    full_review: float
    false_block: float

    def __post_init__(self) -> None:
        if min(self.unsafe_accept, self.short_review, self.full_review, self.false_block) < 0.0:
            raise ValueError("costs cannot be negative")


@dataclass(frozen=True)
class PracticalPolicy:
    schema_version: str
    policy_version: str
    predictive_weights: tuple[float, ...]
    predictive_intercept: float
    route_weights: tuple[float, ...]
    route_intercept: float
    accept_max_risk: float
    short_review_max_risk: float
    full_review_max_risk: float
    calibration_method: str
    calibration_parameters: tuple[float, ...]
    development_sha256: str
    selected_without_test: bool
    false_block_ceiling: float = 0.01
    hard_fault_recall_minimum: float = 0.95

    def __post_init__(self) -> None:
        if self.schema_version != "1.0" or not self.policy_version:
            raise ValueError("unsupported practical policy")
        if len(self.predictive_weights) != 8 or len(self.route_weights) != 10:
            raise ValueError("practical policy feature widths are fixed")
        if self.calibration_method not in {"platt", "isotonic", "temperature", "conformal_selective"}:
            raise ValueError("unsupported calibration method")
        if not self.calibration_parameters:
            raise ValueError("calibration parameters are required")
        if not 0.0 <= self.accept_max_risk <= self.short_review_max_risk <= self.full_review_max_risk <= 1.0:
            raise ValueError("risk thresholds must be monotonic")
        if not SHA256_RE.fullmatch(self.development_sha256):
            raise ValueError("development_sha256 must be a SHA256 digest")
        if not self.selected_without_test:
            raise ValueError("policy selection must not use confirmatory test outcomes")
        if not 0.0 <= self.false_block_ceiling <= 1.0 or not 0.0 <= self.hard_fault_recall_minimum <= 1.0:
            raise ValueError("policy constraints must be in [0, 1]")


@dataclass(frozen=True)
class GuardResult:
    status: HardGuardStatus
    reason_codes: tuple[str, ...]
    missing_evidence: tuple[str, ...]
    fault_source: str | None = None


@dataclass(frozen=True)
class ActionAssessment:
    action: SelectiveAction
    operational_risk: float
    predictive_risk: float
    route_risk: float
    hard_guard_status: HardGuardStatus
    reason_codes: tuple[str, ...]
    missing_evidence: tuple[str, ...]
    review_priority: float
    confidence_interval: tuple[float, float]
    trace_id: str
    model_version: str
    explain_plan_version: str
    policy_version: str
    budget_feasible: bool
    deterministic_replay_sha256: str
    monitoring_events: tuple[str, ...] = field(default_factory=tuple)

    def to_dict(self) -> dict[str, object]:
        payload = asdict(self)
        payload["action"] = self.action.value
        payload["hard_guard_status"] = self.hard_guard_status.value
        return payload


@dataclass(frozen=True)
class BatchAssessment:
    assessments: tuple[ActionAssessment, ...]
    review_budget: float
    realized_review_fraction: float
    budget_feasible: bool
    policy_version: str
    audit_sha256: str


@dataclass(frozen=True)
class PracticalDevelopmentExample:
    object_id: str
    group_id: str
    predictive_features: tuple[float, ...]
    route_features: tuple[float, ...]
    operationally_invalid_action: bool
    partition: str
    source_features_are_oof: bool

    def __post_init__(self) -> None:
        if not self.object_id or not self.group_id:
            raise ValueError("development example identifiers are required")
        if len(self.predictive_features) != 8 or len(self.route_features) != 10:
            raise ValueError("development feature widths are fixed")
        if any(not 0.0 <= value <= 1.0 for value in (*self.predictive_features, *self.route_features)):
            raise ValueError("development features must be normalized")
        if self.partition not in {"train", "validation"}:
            raise ValueError("confirmatory test examples cannot be used for policy development")
        if not self.source_features_are_oof:
            raise ValueError("policy development requires out-of-fold features")


def cost_profile(name: CostProfileName | str) -> CostProfile:
    selected = CostProfileName(name)
    profiles: Mapping[CostProfileName, CostProfile] = {
        CostProfileName.BALANCED: CostProfile(CostProfileName.BALANCED, 8.0, 0.4, 1.0, 3.0),
        CostProfileName.UNSAFE_ACCEPT_SENSITIVE: CostProfile(CostProfileName.UNSAFE_ACCEPT_SENSITIVE, 20.0, 0.5, 1.2, 4.0),
        CostProfileName.REVIEW_EXPENSIVE: CostProfile(CostProfileName.REVIEW_EXPENSIVE, 10.0, 1.5, 4.0, 3.0),
    }
    return profiles[selected]


def ensure_unique_object_ids(items: Sequence[PredictionArtifact]) -> None:
    identifiers = [item.object_id for item in items]
    if len(set(identifiers)) != len(identifiers):
        raise ValueError("batch object_id values must be unique")
