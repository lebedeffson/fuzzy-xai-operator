from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Literal

from fuzzyxai.diagnostics.contracts import canonical_sha256

Action = Literal["ACCEPT", "WARN", "REQUEST_DATA", "REVIEW", "BLOCK"]


@dataclass(frozen=True)
class Evidence:
    evidence_id: str
    kind: str
    value: Any
    source: str
    status: Literal["observed", "missing", "insufficient_evidence"] = "observed"
    artifact_sha256: str = ""
    subject: str = ""
    unit: str | None = None
    sign: int | None = None
    source_version: str = "1.0.0"
    object_id: str = ""
    created_at: str = "2000-01-01T00:00:00Z"

    def __post_init__(self) -> None:
        if self.status == "observed" and self.value is None:
            raise ValueError("observed evidence must carry a value; zero is valid")


@dataclass(frozen=True)
class PredictionRequest:
    scenario_id: str
    object_id: str
    features: dict[str, float]
    controls: dict[str, Any] = field(default_factory=dict)
    requested_view: str = "user"


@dataclass(frozen=True)
class PredictionArtifact:
    object_id: str
    predicted_class: int
    probability: float
    raw_score: float
    model_id: str
    model_version: str
    model_sha256: str
    feature_schema_sha256: str


@dataclass(frozen=True)
class LocalExplanationArtifact:
    object_id: str
    explainer_id: str
    explainer_version: str
    model_version: str
    base_value: float
    shap_values: dict[str, float]
    feature_values: dict[str, float]
    output_sum: float
    output_difference: float
    artifact_sha256: str


@dataclass(frozen=True)
class UncertaintyProfile:
    present_types: tuple[str, ...]
    values: dict[str, float]
    aggregate: float
    conflict: float
    trace_completeness: float


@dataclass(frozen=True)
class RepresentationArtifact:
    representation_id: Literal["F0", "F_int", "NAS", "F_ML"]
    membership: dict[str, Any]
    covered_uncertainties: tuple[str, ...]
    cognitive_complexity: float
    computational_complexity: float
    selection_reason: str


@dataclass(frozen=True)
class ReductionArtifact:
    source_representation: str
    target_representation: str
    reduced_membership: dict[str, float]
    loss: float
    accepted: bool
    maximum_loss: float


@dataclass(frozen=True)
class ObserverDecision:
    action: Action
    risk: float
    reasons: tuple[str, ...]
    critical_issues: tuple[str, ...]


@dataclass(frozen=True)
class ExplanationClaim:
    claim_id: str
    text: str
    evidence_refs: tuple[str, ...]
    status: str
    text_code: str = "registered_template"
    arguments: dict[str, Any] = field(default_factory=dict)
    allowed_views: tuple[str, ...] = ("user", "engineer", "auditor")
    blocked: bool = False


@dataclass(frozen=True)
class ProvenanceRelation:
    source_id: str
    target_id: str
    relation: str
    evidence_ref: str


@dataclass(frozen=True)
class VerticalRun:
    run_id: str
    scenario_id: str
    request: dict[str, Any]
    prediction: dict[str, Any] | None
    explanation: dict[str, Any] | None
    explainable_object: dict[str, Any]
    uncertainty: dict[str, Any]
    representation: dict[str, Any]
    reduction: dict[str, Any]
    route_graph: dict[str, Any]
    diagnosis: dict[str, Any]
    observer: dict[str, Any]
    claims: tuple[dict[str, Any], ...]
    views: dict[str, dict[str, Any]]
    repair: dict[str, Any] | None
    canonical_sha256: str

    @classmethod
    def build(cls, **payload: Any) -> VerticalRun:
        digest_payload = {key: value for key, value in payload.items() if key != "canonical_sha256"}
        return cls(**payload, canonical_sha256=canonical_sha256(digest_payload))


def as_payload(value: Any) -> Any:
    return asdict(value) if hasattr(value, "__dataclass_fields__") else value
