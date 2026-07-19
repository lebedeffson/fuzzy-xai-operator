from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Mapping, Sequence


def _jsonable(value: Any) -> Any:
    if hasattr(value, "tolist"):
        return value.tolist()
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)


class EvidenceContract:
    """Mixin for canonical, JSON-safe evidence contracts."""

    def to_dict(self) -> dict[str, Any]:
        return _jsonable(asdict(self))


@dataclass(frozen=True)
class DataEvidence(EvidenceContract):
    object_id: str
    feature_names: Sequence[str]
    raw_values: Sequence[Any]
    normalized_values: Sequence[float | None]
    missingness: Mapping[str, bool]
    outlier_scores: Mapping[str, float | None]
    anomaly_labels: Sequence[str]
    data_quality: float
    source_trace: Mapping[str, Any]
    warnings: Sequence[str] = field(default_factory=tuple)
    evidence_refs: Sequence[str] = field(default_factory=tuple)


@dataclass(frozen=True)
class TrainingObjectTrace(EvidenceContract):
    object_id: str
    epoch_metrics: Sequence[Mapping[str, Any]]
    predicted_class_by_epoch: Sequence[Any]
    confidence_by_epoch: Sequence[float]
    loss_by_epoch: Sequence[float]
    embedding_by_epoch: Sequence[Sequence[float]]
    rule_activation_by_epoch: Sequence[Mapping[str, float]]
    forgetting_events: Sequence[int]
    stability_score: float
    first_learned_epoch: int | None
    last_correct_epoch: int | None
    warnings: Sequence[str] = field(default_factory=tuple)


@dataclass(frozen=True)
class SubgroupAveragingEvidence(EvidenceContract):
    subgroup_id: str
    size: int
    global_metric_change: float
    subgroup_metric_change: float
    minority_recall_change: float | None
    embedding_collapse: float | None
    prototype_distance_change: float | None
    disappeared_rules: Sequence[str]
    affected_objects: Sequence[str]
    averaged: bool
    limitations: Sequence[str] = field(default_factory=tuple)


@dataclass(frozen=True)
class LearnedRule(EvidenceContract):
    rule_id: str
    model_version: str
    antecedents: Sequence[str]
    consequent: str
    activation: float | None
    coverage: float | None
    precision: float | None
    support: int | None
    stability: float | None
    importance: float | None
    counterfactual_effect: Mapping[str, float]
    source_objects: Sequence[str]
    class_distribution: Mapping[str, float]
    human_text: str
    complexity: float
    is_primary: bool
    is_redundant: bool
    is_conflicting: bool
    native: bool = False
    surrogate: bool = False
    fidelity: float | None = None
    evidence_refs: Sequence[str] = field(default_factory=tuple)


@dataclass(frozen=True)
class ClassConcept(EvidenceContract):
    class_id: str
    class_name: str
    prototype_features: Mapping[str, float]
    prototype_embedding: Sequence[float]
    primary_rules: Sequence[str]
    representative_objects: Sequence[str]
    boundary_objects: Sequence[str]
    counterexamples: Sequence[str]
    intra_class_variability: float | None
    human_description: str
    primary_rule_coverage: float | None
    uncovered_fraction: float | None
    limitations: Sequence[str] = field(default_factory=tuple)


@dataclass(frozen=True)
class SimilarCaseEvidence(EvidenceContract):
    query_object_id: str
    reference_object_id: str
    similarity_score: float
    similarity_method: str
    compared_representation: str
    matched_features: Sequence[str]
    different_features: Sequence[str]
    matched_regions: Sequence[str]
    coverage_score: float | None
    reference_label: Any
    reference_prediction: Any
    reference_outcome: Any
    limitations: Sequence[str]
    trace: Mapping[str, Any]


@dataclass(frozen=True)
class CounterfactualEvidence(EvidenceContract):
    source_prediction: Any
    target_prediction: Any
    changed_features: Mapping[str, Any]
    changed_regions: Sequence[str]
    changed_rules: Sequence[str]
    minimality: float | None
    plausibility: float | None
    stability: float | None
    expected_effect: float | None
    observed_effect: float | None
    actionability: str
    limitations: Sequence[str]
    evidence_refs: Sequence[str] = field(default_factory=tuple)


@dataclass(frozen=True)
class ExplanationNode(EvidenceContract):
    node_id: str
    node_type: str
    label: str
    payload: Mapping[str, Any]
    evidence_refs: Sequence[str] = field(default_factory=tuple)


@dataclass(frozen=True)
class ExplanationEdge(EvidenceContract):
    source: str
    target: str
    relation: str
    evidence_refs: Sequence[str] = field(default_factory=tuple)


@dataclass(frozen=True)
class ExplanationGraph(EvidenceContract):
    nodes: Sequence[ExplanationNode]
    edges: Sequence[ExplanationEdge]
    missing_evidence: Sequence[str] = field(default_factory=tuple)
    schema_version: str = "1.0"


@dataclass(frozen=True)
class HumanExplanation(EvidenceContract):
    level: str
    summary: str
    main_reasons: Sequence[str]
    model_observed: Sequence[str]
    lost_or_averaged: Sequence[str]
    similar_cases: Sequence[str]
    decision_changes: Sequence[str]
    trust: Sequence[str]
    limitations: Sequence[str]
    recommended_action: str
    evidence_trace: Sequence[str]


@dataclass(frozen=True)
class ExplanationEvidence(EvidenceContract):
    data: Sequence[DataEvidence] = field(default_factory=tuple)
    training: Sequence[TrainingObjectTrace] = field(default_factory=tuple)
    subgroups: Sequence[SubgroupAveragingEvidence] = field(default_factory=tuple)
    rules: Sequence[LearnedRule] = field(default_factory=tuple)
    concepts: Sequence[ClassConcept] = field(default_factory=tuple)
    similar_cases: Sequence[SimilarCaseEvidence] = field(default_factory=tuple)
    counterfactuals: Sequence[CounterfactualEvidence] = field(default_factory=tuple)
    missing: Sequence[str] = field(default_factory=tuple)
