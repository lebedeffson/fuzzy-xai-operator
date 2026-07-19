from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Mapping, Sequence

from fuzzyxai.evidence.contracts import ExplanationClaim, ExplanationEvidence, ExplanationGraph


@dataclass(frozen=True)
class StoryStageSpec:
    stage_id: str
    title: str
    status: str
    facts: Sequence[str]
    claim_refs: Sequence[str]


@dataclass(frozen=True)
class FeatureProfileSpec:
    feature: str
    object_value: float | None
    reference_median: float | None
    reference_interval: tuple[float | None, float | None]
    subgroup_interval: tuple[float | None, float | None] | None
    percentile: float | None
    anomaly_status: str
    contribution: float | None
    explanation: str
    claim_refs: Sequence[str]


@dataclass(frozen=True)
class TrainingPointSpec:
    epoch: int
    correct: bool | None
    confidence: float | None
    loss: float | None
    margin: float | None
    prototype_distance: float | None
    rule_activations: Mapping[str, float]
    forgetting: bool


@dataclass(frozen=True)
class TrainingTimelineSpec:
    object_id: str
    points: Sequence[TrainingPointSpec]
    annotations: Sequence[str]
    claim_refs: Sequence[str]


@dataclass(frozen=True)
class KnowledgeRuleSpec:
    rule_id: str
    text: str
    native: bool
    surrogate: bool
    importance: float | None
    coverage: float | None
    precision: float | None
    stability: float | None
    status: str
    counterfactual_effect: Mapping[str, float]
    claim_refs: Sequence[str]


@dataclass(frozen=True)
class ClassConceptSpec:
    class_id: str
    class_name: str
    description: str
    primary_rules: Sequence[str]
    primary_rule_coverage: float | None
    uncovered_fraction: float | None
    representative_objects: Sequence[str]
    boundary_objects: Sequence[str]
    claim_refs: Sequence[str]


@dataclass(frozen=True)
class DecisionEvidenceSpec:
    supports: Sequence[Mapping[str, Any]]
    contradicts: Sequence[Mapping[str, Any]]
    limitations: Sequence[Mapping[str, Any]]


@dataclass(frozen=True)
class SimilarCaseSpec:
    reference_object_id: str
    score: float
    method: str
    representation: str
    matched_features: Sequence[str]
    different_features: Sequence[str]
    limitations: Sequence[str]
    claim_refs: Sequence[str]


@dataclass(frozen=True)
class CounterfactualSpec:
    source_prediction: Any
    target_prediction: Any
    changed_features: Mapping[str, Any]
    changed_rules: Sequence[str]
    observed_effect: float | None
    actionability: str
    limitations: Sequence[str]
    claim_refs: Sequence[str]


@dataclass(frozen=True)
class ProvenanceNodeSpec:
    node_id: str
    node_type: str
    label: str


@dataclass(frozen=True)
class ProvenanceEdgeSpec:
    source: str
    target: str
    relation: str


@dataclass(frozen=True)
class ExplanationVisualSpec:
    """Typed backend-neutral presentation contract for explanation views."""

    overview: Mapping[str, Any]
    story: Sequence[StoryStageSpec]
    data_profile: Sequence[FeatureProfileSpec]
    training_timeline: Sequence[TrainingTimelineSpec]
    knowledge_atlas: Mapping[str, Any]
    decision_evidence: DecisionEvidenceSpec
    similar_cases: Sequence[SimilarCaseSpec]
    counterfactuals: Sequence[CounterfactualSpec]
    provenance_nodes: Sequence[ProvenanceNodeSpec]
    provenance_edges: Sequence[ProvenanceEdgeSpec]
    schema_version: str = "1.0"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _claim_refs(claims: Sequence[ExplanationClaim], claim_type: str, subject_id: str | None = None) -> list[str]:
    return [
        claim.claim_id
        for claim in claims
        if claim.claim_type == claim_type and (subject_id is None or claim.subject_id == subject_id)
    ]


def _claim_item(claim: ExplanationClaim) -> dict[str, Any]:
    return {
        "claim_id": claim.claim_id,
        "statement": claim.short_statement,
        "status": claim.status,
        "strength": claim.strength,
        "limitations": list(claim.limitations),
    }


def build_visual_spec(
    evidence: ExplanationEvidence,
    claims: Sequence[ExplanationClaim],
    graph: ExplanationGraph,
    *,
    prediction: Mapping[str, Any],
    action: str,
    contributions: Mapping[str, float] | None = None,
    explanation_level: Mapping[str, Any] | None = None,
) -> ExplanationVisualSpec:
    contributions = dict(contributions or {})
    claim_types = {claim.claim_type for claim in claims}

    def stage(stage_id: str, title: str, types: set[str]) -> StoryStageSpec:
        selected = [claim for claim in claims if claim.claim_type in types]
        if not selected:
            status = "missing"
        elif any(claim.status == "contested" for claim in selected):
            status = "conflict"
        elif any(claim.status == "insufficient_evidence" for claim in selected):
            status = "limitation"
        else:
            status = "supported"
        return StoryStageSpec(
            stage_id=stage_id,
            title=title,
            status=status,
            facts=tuple(claim.short_statement for claim in selected[:4]),
            claim_refs=tuple(claim.claim_id for claim in selected),
        )

    story = (
        stage("data", "Данные", {"data_quality", "data_deviation", "data_error_status"}),
        stage("training", "Обучение", {"first_learned", "forgetting", "subgroup_averaging", "lost_rules"}),
        stage("knowledge", "Знания модели", {"model_rule", "class_concept"}),
        stage("decision", "Решение", {"prediction", "similar_case", "counterfactual", "diagnostic"}),
        stage("action", "Действие", {"recommended_action"}),
    )

    profiles: list[FeatureProfileSpec] = []
    for item in evidence.data[:1]:
        raw = dict(zip(item.feature_names, item.raw_values))
        for feature in item.feature_names:
            profile = dict(item.reference_profiles.get(feature, {}))
            value = raw.get(feature)
            missing = item.missingness.get(feature, False)
            anomalous = feature in item.anomaly_labels
            status = "missing" if missing else ("deviation_not_error" if anomalous else "within_reference")
            refs = [
                claim.claim_id
                for claim in claims
                if claim.subject_id == item.object_id and claim.claim_type in {"data_quality", "data_deviation", "data_error_status"}
            ]
            explanation = (
                "Значение отсутствует."
                if missing
                else "Значение отклоняется от общего референса; это не доказательство ошибки."
                if anomalous
                else "Значение находится в наблюдаемом референсном профиле."
            )
            subgroup = item.subgroup_profiles.get(feature)
            profiles.append(
                FeatureProfileSpec(
                    feature=feature,
                    object_value=float(value) if isinstance(value, (int, float)) else None,
                    reference_median=profile.get("median"),
                    reference_interval=(profile.get("q05"), profile.get("q95")),
                    subgroup_interval=(subgroup.get("q05"), subgroup.get("q95")) if subgroup else None,
                    percentile=profile.get("percentile"),
                    anomaly_status=status,
                    contribution=float(contributions[feature]) if feature in contributions else None,
                    explanation=explanation,
                    claim_refs=tuple(refs),
                )
            )

    timelines: list[TrainingTimelineSpec] = []
    for trace in evidence.training:
        points: list[TrainingPointSpec] = []
        for index, metric in enumerate(trace.epoch_metrics):
            epoch = int(metric.get("epoch", index))
            points.append(
                TrainingPointSpec(
                    epoch=epoch,
                    correct=bool(metric["correct"]) if "correct" in metric else None,
                    confidence=float(metric["confidence"]) if metric.get("confidence") is not None else None,
                    loss=float(metric["loss"]) if metric.get("loss") is not None else None,
                    margin=float(metric["margin"]) if metric.get("margin") is not None else None,
                    prototype_distance=float(metric["prototype_distance"]) if metric.get("prototype_distance") is not None else None,
                    rule_activations={str(key): float(value) for key, value in dict(metric.get("rule_activations", {})).items()},
                    forgetting=epoch in trace.forgetting_events,
                )
            )
        refs = _claim_refs(claims, "first_learned", trace.object_id) + _claim_refs(claims, "forgetting", trace.object_id)
        annotations = [claim.short_statement for claim in claims if claim.claim_id in refs]
        timelines.append(TrainingTimelineSpec(trace.object_id, points, annotations, refs))

    rules = [
        KnowledgeRuleSpec(
            rule_id=rule.rule_id,
            text=rule.human_text,
            native=rule.native,
            surrogate=rule.surrogate,
            importance=rule.importance,
            coverage=rule.coverage,
            precision=rule.precision,
            stability=rule.stability,
            status="conflict" if rule.is_conflicting else "stable" if rule.stability is not None else "unmeasured",
            counterfactual_effect=dict(rule.counterfactual_effect),
            claim_refs=tuple(_claim_refs(claims, "model_rule", rule.rule_id)),
        )
        for rule in evidence.rules
    ]
    concepts = [
        ClassConceptSpec(
            class_id=concept.class_id,
            class_name=concept.class_name,
            description=concept.human_description,
            primary_rules=tuple(concept.primary_rules),
            primary_rule_coverage=concept.primary_rule_coverage,
            uncovered_fraction=concept.uncovered_fraction,
            representative_objects=tuple(concept.representative_objects),
            boundary_objects=tuple(concept.boundary_objects),
            claim_refs=tuple(_claim_refs(claims, "class_concept", concept.class_id)),
        )
        for concept in evidence.concepts
    ]

    support_types = {"prediction", "model_rule", "class_concept", "similar_case"}
    contradict_types = {"data_deviation", "forgetting", "subgroup_averaging", "lost_rules", "diagnostic"}
    supports = [_claim_item(claim) for claim in claims if claim.claim_type in support_types and claim.status == "supported"]
    contradicts = [_claim_item(claim) for claim in claims if claim.claim_type in contradict_types and claim.status != "insufficient_evidence"]
    limitations = [_claim_item(claim) for claim in claims if claim.status in {"contested", "insufficient_evidence"}]

    similar_specs = [
        SimilarCaseSpec(
            reference_object_id=item.reference_object_id,
            score=item.similarity_score,
            method=item.similarity_method,
            representation=item.compared_representation,
            matched_features=tuple(item.matched_features),
            different_features=tuple(item.different_features),
            limitations=tuple(item.limitations),
            claim_refs=tuple(_claim_refs(claims, "similar_case", item.query_object_id)),
        )
        for item in evidence.similar_cases
    ]
    counterfactual_specs = [
        CounterfactualSpec(
            source_prediction=item.source_prediction,
            target_prediction=item.target_prediction,
            changed_features=dict(item.changed_features),
            changed_rules=tuple(item.changed_rules),
            observed_effect=item.observed_effect,
            actionability=item.actionability,
            limitations=tuple(item.limitations),
            claim_refs=tuple(_claim_refs(claims, "counterfactual", str(item.source_prediction))),
        )
        for item in evidence.counterfactuals
    ]

    overview = {
        "prediction": prediction,
        "action": action,
        "explanation_level": dict(explanation_level or {}),
        "claim_count": len(claims),
        "supported_claims": sum(claim.status == "supported" for claim in claims),
        "contested_claims": sum(claim.status == "contested" for claim in claims),
        "insufficient_claims": sum(claim.status == "insufficient_evidence" for claim in claims),
        "available_claim_types": sorted(claim_types),
    }
    return ExplanationVisualSpec(
        overview=overview,
        story=story,
        data_profile=profiles,
        training_timeline=timelines,
        knowledge_atlas={"rules": [asdict(rule) for rule in rules], "concepts": [asdict(item) for item in concepts]},
        decision_evidence=DecisionEvidenceSpec(supports, contradicts, limitations),
        similar_cases=similar_specs,
        counterfactuals=counterfactual_specs,
        provenance_nodes=[ProvenanceNodeSpec(node.node_id, node.node_type, node.label) for node in graph.nodes],
        provenance_edges=[ProvenanceEdgeSpec(edge.source, edge.target, edge.relation) for edge in graph.edges],
    )
