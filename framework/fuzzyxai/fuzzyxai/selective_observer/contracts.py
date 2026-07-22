"""Typed contracts for the two-stage selective-observer research cycle."""

from __future__ import annotations

import re
from dataclasses import asdict, dataclass
from enum import Enum
from typing import Sequence


SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
COMMIT_RE = re.compile(r"^[0-9a-f]{40}$")


class SelectiveAction(str, Enum):
    ACCEPT = "accept"
    SHORT_REVIEW = "short_review"
    FULL_REVIEW = "full_review"
    REPAIR_THEN_RETRY = "repair_then_retry"
    BLOCK = "block"


class ResearchPartition(str, Enum):
    TRAIN = "train"
    VALIDATION = "validation"
    TEST = "test"


FEATURE_NAMES = (
    "model_uncertainty",
    "calibration_residual",
    "boundary_proximity",
    "model_disagreement",
    "explainer_disagreement",
    "attribution_instability",
    "provenance_incompleteness",
    "data_shift",
    "representation_loss",
    "rupture_severity",
    "rare_group",
)


@dataclass(frozen=True)
class SelectiveRiskFeatures:
    """Validation-derived risk channels normalized to the closed interval [0, 1]."""

    model_uncertainty: float
    calibration_residual: float
    boundary_proximity: float
    model_disagreement: float
    explainer_disagreement: float
    attribution_instability: float
    provenance_incompleteness: float
    data_shift: float
    representation_loss: float
    rupture_severity: float
    rare_group: float

    def __post_init__(self) -> None:
        for name, value in zip(FEATURE_NAMES, self.to_vector()):
            if not 0.0 <= value <= 1.0:
                raise ValueError(f"{name} must be in [0, 1]")

    def to_vector(self) -> tuple[float, ...]:
        return tuple(float(getattr(self, name)) for name in FEATURE_NAMES)


@dataclass(frozen=True)
class DevelopmentExample:
    object_id: str
    features: SelectiveRiskFeatures
    unsafe_automatic_action: bool
    partition: ResearchPartition
    source_features_are_oof: bool
    group_id: str

    def __post_init__(self) -> None:
        if self.partition is ResearchPartition.TEST:
            raise ValueError("controller development cannot consume the confirmatory test partition")
        if not self.source_features_are_oof:
            raise ValueError("controller development requires out-of-fold source features")
        if not self.object_id or not self.group_id:
            raise ValueError("object_id and group_id are required")


@dataclass(frozen=True)
class ConfirmatoryExample:
    object_id: str
    features: SelectiveRiskFeatures
    unsafe_automatic_action: bool
    partition: ResearchPartition = ResearchPartition.TEST

    def __post_init__(self) -> None:
        if self.partition is not ResearchPartition.TEST:
            raise ValueError("confirmatory examples must come from the frozen test partition")


@dataclass(frozen=True)
class SelectiveControllerSpec:
    """Serializable controller frozen before confirmatory test access."""

    schema_version: str
    feature_names: tuple[str, ...]
    coefficients: tuple[float, ...]
    intercept: float
    feature_means: tuple[float, ...]
    feature_scales: tuple[float, ...]
    accept_max_risk: float
    short_review_max_risk: float
    full_review_max_risk: float
    block_rupture_severity: float
    development_hash: str
    selected_without_test: bool

    def __post_init__(self) -> None:
        width = len(self.feature_names)
        if self.schema_version != "1.0":
            raise ValueError("SelectiveControllerSpec requires schema version 1.0")
        if self.feature_names != FEATURE_NAMES:
            raise ValueError("controller feature order is not canonical")
        if any(len(values) != width for values in (self.coefficients, self.feature_means, self.feature_scales)):
            raise ValueError("controller vectors have inconsistent widths")
        if not 0.0 <= self.accept_max_risk <= self.short_review_max_risk <= self.full_review_max_risk <= 1.0:
            raise ValueError("controller thresholds must be monotonic")
        if not 0.0 <= self.block_rupture_severity <= 1.0:
            raise ValueError("block rupture threshold must be in [0, 1]")
        if not SHA256_RE.fullmatch(self.development_hash):
            raise ValueError("development_hash must be a SHA256 digest")
        if not self.selected_without_test:
            raise ValueError("a confirmatory controller cannot be selected using test outcomes")
        if any(value <= 0.0 for value in self.feature_scales):
            raise ValueError("feature scales must be positive")

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True)
class PolicyMetrics:
    n_objects: int
    coverage: float
    selective_risk: float
    wrong_automatic: int
    short_review: int
    full_review: int
    blocked: int
    manual_review_fraction: float


@dataclass(frozen=True)
class RouteContractRecord:
    object_id: str
    mandatory_nodes: tuple[str, ...]
    observed_nodes: tuple[str, ...]
    fault_type: str | None
    fault_source: str | None
    detected_fault_type: str | None
    detected_fault_source: str | None
    rupture_severity: float
    requested_action: SelectiveAction
    model_error: bool

    def __post_init__(self) -> None:
        if not self.object_id or not self.mandatory_nodes:
            raise ValueError("route record requires an object and mandatory nodes")
        if not 0.0 <= self.rupture_severity <= 1.0:
            raise ValueError("rupture severity must be in [0, 1]")

    @property
    def contract_violated(self) -> bool:
        return bool(set(self.mandatory_nodes) - set(self.observed_nodes)) or self.fault_type is not None

    @property
    def invalid_automatic_action(self) -> bool:
        return self.requested_action is SelectiveAction.ACCEPT and self.contract_violated


@dataclass(frozen=True)
class PredictiveRouteExample:
    object_id: str
    baseline_features: tuple[float, ...]
    typed_route_features: tuple[float, ...]
    model_error: bool
    partition: ResearchPartition
    source_features_are_oof: bool

    def __post_init__(self) -> None:
        if not self.baseline_features or not self.typed_route_features:
            raise ValueError("both baseline and typed route features are required")
        if self.partition is not ResearchPartition.TEST and not self.source_features_are_oof:
            raise ValueError("development route features must be out-of-fold")


@dataclass(frozen=True)
class RuleProfile:
    rule_id: str
    subgroup_specificity: float
    subgroup_coverage: float
    activation_stability: float
    functional_redundancy: float
    unique_prediction_fraction: float
    margin_contribution: float
    depth: int
    length: int
    selection_partition: ResearchPartition

    def __post_init__(self) -> None:
        channels: Sequence[float] = (
            self.subgroup_specificity,
            self.subgroup_coverage,
            self.activation_stability,
            self.functional_redundancy,
            self.unique_prediction_fraction,
            self.margin_contribution,
        )
        if not self.rule_id or any(not 0.0 <= value <= 1.0 for value in channels):
            raise ValueError("rule profile channels must be identified and normalized")
        if self.depth <= 0 or self.length <= 0:
            raise ValueError("rule depth and length must be positive")
        if self.selection_partition is ResearchPartition.TEST:
            raise ValueError("rules cannot be selected on the confirmatory test partition")


@dataclass(frozen=True)
class RuleAblationObservation:
    dataset_id: str
    candidate_rule_id: str
    candidate_effect: float
    matched_control_effects: tuple[float, ...]
    partition: ResearchPartition = ResearchPartition.TEST

    def __post_init__(self) -> None:
        if self.partition is not ResearchPartition.TEST:
            raise ValueError("confirmatory ablation observations must be held out")
        if len(self.matched_control_effects) < 5:
            raise ValueError("each candidate requires at least five matched controls")


@dataclass(frozen=True)
class ConfirmatoryProtocolLock:
    """Immutable stage-B declaration required before any confirmatory opening."""

    schema_version: str
    frozen_predecessor_commit: str
    implementation_commit: str
    interface_sha256: str
    dictionary_sha256: str
    formative_dataset_hashes: tuple[str, ...]
    confirmatory_dataset_hashes: tuple[str, ...]
    formative_participant_hashes: tuple[str, ...]
    confirmatory_participant_hashes: tuple[str, ...]
    primary_outcomes: tuple[str, ...]
    preregistered_baselines: tuple[str, ...]
    minimum_effects: tuple[str, ...]
    statistical_tests: tuple[str, ...]
    exclusion_rules: tuple[str, ...]
    stopping_rule: str
    required_sample_size: int
    independent_timestamp: str
    protocol_sha256: str

    def __post_init__(self) -> None:
        if self.schema_version != "1.0":
            raise ValueError("ConfirmatoryProtocolLock requires schema version 1.0")
        if not COMMIT_RE.fullmatch(self.frozen_predecessor_commit) or not COMMIT_RE.fullmatch(self.implementation_commit):
            raise ValueError("protocol commits must be complete Git hashes")
        hashes = (
            self.interface_sha256,
            self.dictionary_sha256,
            self.protocol_sha256,
            *self.formative_dataset_hashes,
            *self.confirmatory_dataset_hashes,
            *self.formative_participant_hashes,
            *self.confirmatory_participant_hashes,
        )
        if any(not SHA256_RE.fullmatch(value) for value in hashes):
            raise ValueError("protocol identities must use complete SHA256 hashes")
        if set(self.formative_dataset_hashes) & set(self.confirmatory_dataset_hashes):
            raise ValueError("confirmatory datasets overlap formative datasets")
        if set(self.formative_participant_hashes) & set(self.confirmatory_participant_hashes):
            raise ValueError("confirmatory participants overlap formative participants")
        required = (
            self.primary_outcomes,
            self.preregistered_baselines,
            self.minimum_effects,
            self.statistical_tests,
            self.exclusion_rules,
        )
        if any(not values for values in required) or not self.stopping_rule or not self.independent_timestamp:
            raise ValueError("confirmatory lock is incomplete")
        if self.required_sample_size <= 0:
            raise ValueError("power analysis must provide a positive sample size")
