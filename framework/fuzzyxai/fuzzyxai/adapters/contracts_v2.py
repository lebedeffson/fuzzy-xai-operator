from __future__ import annotations

from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import Any, Literal, Mapping, Sequence

from .model import AdapterCapabilities


class TaskType(str, Enum):
    BINARY_CLASSIFICATION = "binary_classification"
    MULTICLASS_CLASSIFICATION = "multiclass_classification"
    MULTILABEL_CLASSIFICATION = "multilabel_classification"
    REGRESSION = "regression"
    ANOMALY_DETECTION = "anomaly_detection"
    CLUSTERING = "clustering"
    FORECASTING = "forecasting"
    RANKING = "ranking"
    TEXT_CLASSIFICATION = "text_classification"
    IMAGE_CLASSIFICATION = "image_classification"


EvidenceOrigin = Literal["native", "derived", "derived_from_native", "surrogate", "external"]
FidelityStatus = Literal["not_applicable", "not_measured", "measured", "insufficient"]


@dataclass(frozen=True)
class EvidenceChannelDescriptor:
    name: str
    available: bool
    origin: EvidenceOrigin
    method: str
    fidelity_status: FidelityStatus = "not_applicable"
    fidelity: float | None = None
    limitations: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if not self.available and self.fidelity is not None:
            raise ValueError("unavailable evidence channel cannot have fidelity")
        if self.origin == "surrogate" and self.available and self.fidelity_status == "not_applicable":
            raise ValueError("surrogate evidence must disclose fidelity status")
        if self.fidelity is not None and not 0.0 <= self.fidelity <= 1.0:
            raise ValueError("fidelity must be in [0, 1]")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class ModelCapabilities(AdapterCapabilities):
    decision_function: bool = False
    uncertainty: bool = False
    calibration: bool = False
    local_contributions: bool = False
    global_importance: bool = False
    native_rules: bool = False
    decision_path: bool = False
    support_examples: bool = False
    nearest_neighbors: bool = False
    integrated_gradients: bool = False
    attention: bool = False
    occlusion: bool = False
    per_sample_loss: bool = False
    native_counterfactual: bool = False
    perturbation_counterfactual: bool = True
    feature_names: bool = False
    class_names: bool = False
    input_constraints: bool = False
    channels: tuple[EvidenceChannelDescriptor, ...] = ()

    def get(self, name: str, default: bool = False) -> bool:
        aliases = {"feature_importance": "global_importance", "rules": "native_rules"}
        return bool(getattr(self, aliases.get(name, name), default))

    def descriptor(self, name: str) -> EvidenceChannelDescriptor | None:
        return next((item for item in self.channels if item.name == name), None)

    def available_channels(self) -> tuple[str, ...]:
        declared = [item.name for item in self.channels if item.available]
        return tuple(dict.fromkeys(declared))

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["feature_importance"] = self.global_importance
        payload["rules"] = self.native_rules
        return payload


@dataclass(frozen=True)
class ModelInputSchema:
    shape: tuple[int | None, ...] = ()
    dtype: str = "unknown"
    feature_names: tuple[str, ...] = ()
    transformed_feature_names: tuple[str, ...] = ()
    feature_provenance: Mapping[str, tuple[str, ...]] = field(default_factory=dict)
    constraints: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class ModelOutputSchema:
    task_type: TaskType
    classes: tuple[Any, ...] = ()
    output_names: tuple[str, ...] = ()
    score_semantics: str = "model_output"
    calibrated_probability: bool = False


@dataclass(frozen=True)
class ExplanationContext:
    reference_data: Any = None
    reference_labels: Any = None
    feature_names: tuple[str, ...] = ()
    target: Any = None
    budget: str = "standard"
    requested_channels: tuple[str, ...] = ()
    input_constraints: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class LocalModelEvidence:
    channels: Mapping[str, Any] = field(default_factory=dict)
    descriptors: tuple[EvidenceChannelDescriptor, ...] = ()
    missing_channels: tuple[str, ...] = ()
    limitations: tuple[str, ...] = ()

    def to_runtime_mapping(self) -> dict[str, Any]:
        payload = dict(self.channels)
        payload["evidence_descriptors"] = [item.to_dict() for item in self.descriptors]
        payload["missing_channels"] = list(self.missing_channels)
        payload["limitations"] = list(self.limitations)
        return payload


@dataclass(frozen=True)
class GlobalModelEvidence:
    channels: Mapping[str, Any] = field(default_factory=dict)
    descriptors: tuple[EvidenceChannelDescriptor, ...] = ()
    limitations: tuple[str, ...] = ()


@dataclass(frozen=True)
class AdapterResolutionReport:
    selected_adapter: str
    selected_family: str
    task_type: TaskType
    matched_predicates: tuple[str, ...]
    rejected_adapters: tuple[str, ...]
    warnings: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["task_type"] = self.task_type.value
        return payload


@dataclass(frozen=True)
class AdapterCheck:
    check_id: str
    status: Literal["pass", "fail", "not_applicable"]
    detail: str


@dataclass(frozen=True)
class AdapterConformanceReport:
    adapter_id: str
    model_family: str
    task_type: TaskType
    status: Literal["pass", "fail"]
    checks: tuple[AdapterCheck, ...]
    errors: tuple[str, ...] = ()
    warnings: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["task_type"] = self.task_type.value
        return payload


@dataclass(frozen=True)
class ExplanationPlanDecision:
    selected_channels: tuple[str, ...]
    skipped_channels: Mapping[str, str]
    required_quality_checks: tuple[str, ...]
    limitations: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def serialize_descriptors(values: Sequence[EvidenceChannelDescriptor]) -> list[dict[str, Any]]:
    return [item.to_dict() for item in values]
