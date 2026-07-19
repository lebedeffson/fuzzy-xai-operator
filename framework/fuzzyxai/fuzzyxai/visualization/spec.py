from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from typing import Any, Mapping, Sequence

from fuzzyxai.evidence.contracts import ExplanationClaim, ExplanationEvidence, ExplanationGraph


@dataclass(frozen=True)
class PredictionVisualSpec:
    prediction: str
    score: float | None
    model_type: str
    adapter_id: str


@dataclass(frozen=True)
class ExplanationLevelVisualSpec:
    level: str
    rationale: str
    available_channels: tuple[str, ...]
    missing_channels: tuple[str, ...]
    native_channels: tuple[str, ...]
    surrogate_channels: tuple[str, ...]


@dataclass(frozen=True)
class ClaimCountSpec:
    total: int
    supported: int
    contested: int
    insufficient_evidence: int
    not_applicable: int


@dataclass(frozen=True)
class OverviewSpec:
    prediction: PredictionVisualSpec
    action: str
    explanation_level: ExplanationLevelVisualSpec
    claim_counts: ClaimCountSpec
    available_claim_types: tuple[str, ...]


@dataclass(frozen=True)
class StoryStageSpec:
    stage_id: str
    title: str
    evidence_status: str
    effect: str
    severity: str
    facts: tuple[str, ...]
    claim_refs: tuple[str, ...]


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
    claim_refs: tuple[str, ...]


@dataclass(frozen=True)
class RuleActivationSpec:
    rule_id: str
    activation: float


@dataclass(frozen=True)
class TrainingPointSpec:
    epoch: int
    correct: bool | None
    confidence: float | None
    loss: float | None
    margin: float | None
    prototype_distance: float | None
    global_metric: float | None
    subgroup_metric: float | None
    rule_activations: tuple[RuleActivationSpec, ...]
    forgetting: bool


@dataclass(frozen=True)
class TrainingTimelineSpec:
    object_id: str
    points: tuple[TrainingPointSpec, ...]
    annotations: tuple[str, ...]
    claim_refs: tuple[str, ...]


@dataclass(frozen=True)
class MetricValueSpec:
    metric: str
    value: float


@dataclass(frozen=True)
class RuleEffectSpec:
    train: float | None
    validation: float | None
    test: float | None
    subgroup_recall: float | None
    critical_errors: float | None
    calibration: float | None
    additional: tuple[MetricValueSpec, ...]


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
    counterfactual_effect: RuleEffectSpec
    claim_refs: tuple[str, ...]


@dataclass(frozen=True)
class ClassConceptSpec:
    class_id: str
    class_name: str
    description: str
    primary_rules: tuple[str, ...]
    primary_rule_coverage: float | None
    uncovered_fraction: float | None
    representative_objects: tuple[str, ...]
    boundary_objects: tuple[str, ...]
    claim_refs: tuple[str, ...]


@dataclass(frozen=True)
class KnowledgeAtlasSpec:
    rules: tuple[KnowledgeRuleSpec, ...]
    concepts: tuple[ClassConceptSpec, ...]
    source_rule_count: int
    displayed_rule_count: int
    primary_rule_count: int


@dataclass(frozen=True)
class ClaimVisualSpec:
    claim_id: str
    statement: str
    evidence_status: str
    effect: str
    severity: str
    strength: float | None
    evidence_refs: tuple[str, ...]
    limitations: tuple[str, ...]
    native: bool | None
    surrogate: bool | None


@dataclass(frozen=True)
class DecisionEvidenceSpec:
    supports: tuple[ClaimVisualSpec, ...]
    contradicts: tuple[ClaimVisualSpec, ...]
    limitations: tuple[ClaimVisualSpec, ...]


@dataclass(frozen=True)
class SimilarCaseSpec:
    query_object_id: str
    reference_object_id: str
    score: float
    method: str
    representation: str
    matched_features: tuple[str, ...]
    different_features: tuple[str, ...]
    matched_regions: tuple[str, ...]
    coverage_score: float | None
    reference_label: str
    reference_prediction: str
    is_counterexample: bool
    media_artifacts: tuple[tuple[str, str], ...]
    limitations: tuple[str, ...]
    claim_refs: tuple[str, ...]


@dataclass(frozen=True)
class FeatureChangeSpec:
    feature: str
    source_value: str | None
    target_value: str


@dataclass(frozen=True)
class CounterfactualSpec:
    source_prediction: str
    target_prediction: str
    changed_features: tuple[FeatureChangeSpec, ...]
    changed_rules: tuple[str, ...]
    minimality: float | None
    plausibility: float | None
    expected_effect: float | None
    observed_effect: float | None
    actionability: str
    limitations: tuple[str, ...]
    claim_refs: tuple[str, ...]


@dataclass(frozen=True)
class RuleAblationMetricSpec:
    rule_id: str
    metric: str
    with_rule: float
    without_rule: float
    difference: float
    scope: str


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
class AuditSpec:
    graph_valid: bool
    graph_errors: tuple[str, ...]
    node_count: int
    edge_count: int
    claim_count: int
    missing_evidence: tuple[str, ...]


@dataclass(frozen=True)
class ExplanationVisualSpec:
    """Strict backend-neutral presentation contract for all renderers."""

    overview: OverviewSpec
    story: tuple[StoryStageSpec, ...]
    data_profile: tuple[FeatureProfileSpec, ...]
    training_timeline: tuple[TrainingTimelineSpec, ...]
    knowledge_atlas: KnowledgeAtlasSpec
    decision_evidence: DecisionEvidenceSpec
    similar_cases: tuple[SimilarCaseSpec, ...]
    counterfactuals: tuple[CounterfactualSpec, ...]
    rule_ablations: tuple[RuleAblationMetricSpec, ...]
    provenance_nodes: tuple[ProvenanceNodeSpec, ...]
    provenance_edges: tuple[ProvenanceEdgeSpec, ...]
    audit: AuditSpec
    schema_version: str = "1.1"

    def to_dict(self) -> dict[str, object]:
        return asdict(self)

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "ExplanationVisualSpec":
        return visual_spec_from_dict(payload)


def migrate_visual_spec(payload: Mapping[str, Any]) -> dict[str, Any]:
    """Migrate the legacy 1.0 display payload without inventing missing evidence."""

    data = dict(payload)
    if str(data.get("schema_version", "1.0")) == "1.1":
        return data
    overview = dict(data.get("overview", {}))
    level = dict(overview.get("explanation_level", {}))
    prediction = dict(overview.get("prediction", {}))
    data["overview"] = {
        "prediction": {
            "prediction": _text(prediction.get("predictions")),
            "score": _number(prediction.get("score")),
            "model_type": str(prediction.get("model_type", "unknown")),
            "adapter_id": str(prediction.get("adapter_id", "unknown")),
        },
        "action": str(overview.get("action", "review")),
        "explanation_level": {
            "level": str(level.get("level", "E0")),
            "rationale": str(level.get("rationale", "legacy VisualSpec")),
            "available_channels": list(level.get("available_channels", ())),
            "missing_channels": list(level.get("missing_channels", ())),
            "native_channels": list(level.get("native_channels", ())),
            "surrogate_channels": list(level.get("surrogate_channels", ())),
        },
        "claim_counts": {
            "total": int(overview.get("claim_count", 0)),
            "supported": int(overview.get("supported_claims", 0)),
            "contested": int(overview.get("contested_claims", 0)),
            "insufficient_evidence": int(overview.get("insufficient_claims", 0)),
            "not_applicable": 0,
        },
        "available_claim_types": list(overview.get("available_claim_types", ())),
    }
    migrated_story = []
    for item in data.get("story", ()):
        stage = dict(item)
        legacy_status = str(stage.pop("status", "missing"))
        stage.setdefault("evidence_status", "supported" if legacy_status == "supported" else "contested" if legacy_status == "conflict" else "insufficient_evidence")
        stage.setdefault("effect", "adverse" if legacy_status == "conflict" else "unknown" if legacy_status == "missing" else "neutral")
        stage.setdefault("severity", "warning" if legacy_status != "supported" else "info")
        migrated_story.append(stage)
    data["story"] = migrated_story
    atlas = dict(data.get("knowledge_atlas", {}))
    atlas.setdefault("source_rule_count", len(atlas.get("rules", ())))
    atlas.setdefault("displayed_rule_count", len(atlas.get("rules", ())))
    atlas.setdefault("primary_rule_count", 0)
    data["knowledge_atlas"] = atlas
    migrated_counterfactuals = []
    for item in data.get("counterfactuals", ()):
        counterfactual = dict(item)
        changes = counterfactual.get("changed_features", {})
        if isinstance(changes, Mapping):
            counterfactual["changed_features"] = [
                {"feature": str(name), "source_value": None, "target_value": _text(value)}
                for name, value in changes.items()
            ]
        migrated_counterfactuals.append(counterfactual)
    data["counterfactuals"] = migrated_counterfactuals
    data.setdefault("rule_ablations", [])
    data.setdefault(
        "audit",
        {
            "graph_valid": False,
            "graph_errors": ["legacy VisualSpec was migrated but not graph-revalidated"],
            "node_count": len(data.get("provenance_nodes", ())),
            "edge_count": len(data.get("provenance_edges", ())),
            "claim_count": data["overview"]["claim_counts"]["total"],
            "missing_evidence": [],
        },
    )
    data["schema_version"] = "1.1"
    return data


def visual_spec_from_dict(payload: Mapping[str, Any]) -> ExplanationVisualSpec:
    data = migrate_visual_spec(payload)
    overview = dict(data["overview"])
    prediction = dict(overview["prediction"])
    level = dict(overview["explanation_level"])
    counts = dict(overview["claim_counts"])

    def claim_item(item: Mapping[str, Any]) -> ClaimVisualSpec:
        return ClaimVisualSpec(
            claim_id=str(item["claim_id"]), statement=str(item["statement"]),
            evidence_status=str(item.get("evidence_status", item.get("status", "insufficient_evidence"))),
            effect=str(item.get("effect", "unknown")), severity=str(item.get("severity", "warning")),
            strength=_number(item.get("strength")), evidence_refs=tuple(str(value) for value in item.get("evidence_refs", ())),
            limitations=tuple(str(value) for value in item.get("limitations", ())),
            native=None if item.get("native") is None else bool(item["native"]),
            surrogate=None if item.get("surrogate") is None else bool(item["surrogate"]),
        )

    atlas = dict(data.get("knowledge_atlas", {}))
    rules = []
    for item in atlas.get("rules", ()):
        rule = dict(item)
        effects = dict(rule.get("counterfactual_effect", {}))
        additional = effects.pop("additional", ())
        rules.append(KnowledgeRuleSpec(
            rule_id=str(rule["rule_id"]), text=str(rule.get("text", "")), native=bool(rule.get("native")), surrogate=bool(rule.get("surrogate")),
            importance=_number(rule.get("importance")), coverage=_number(rule.get("coverage")), precision=_number(rule.get("precision")), stability=_number(rule.get("stability")), status=str(rule.get("status", "unmeasured")),
            counterfactual_effect=RuleEffectSpec(
                train=_number(effects.get("train")), validation=_number(effects.get("validation")), test=_number(effects.get("test")),
                subgroup_recall=_number(effects.get("subgroup_recall")), critical_errors=_number(effects.get("critical_errors")), calibration=_number(effects.get("calibration")),
                additional=tuple(MetricValueSpec(str(value["metric"]), float(value["value"])) for value in additional),
            ), claim_refs=tuple(str(value) for value in rule.get("claim_refs", ())),
        ))
    concepts = tuple(ClassConceptSpec(
        class_id=str(item["class_id"]), class_name=str(item.get("class_name", "")), description=str(item.get("description", "")),
        primary_rules=tuple(str(value) for value in item.get("primary_rules", ())), primary_rule_coverage=_number(item.get("primary_rule_coverage")), uncovered_fraction=_number(item.get("uncovered_fraction")),
        representative_objects=tuple(str(value) for value in item.get("representative_objects", ())), boundary_objects=tuple(str(value) for value in item.get("boundary_objects", ())), claim_refs=tuple(str(value) for value in item.get("claim_refs", ())),
    ) for item in atlas.get("concepts", ()))
    decision = dict(data.get("decision_evidence", {}))
    audit = dict(data.get("audit", {}))
    return ExplanationVisualSpec(
        overview=OverviewSpec(
            prediction=PredictionVisualSpec(str(prediction.get("prediction", "")), _number(prediction.get("score")), str(prediction.get("model_type", "unknown")), str(prediction.get("adapter_id", "unknown"))),
            action=str(overview.get("action", "review")),
            explanation_level=ExplanationLevelVisualSpec(str(level.get("level", "E0")), str(level.get("rationale", "")), tuple(level.get("available_channels", ())), tuple(level.get("missing_channels", ())), tuple(level.get("native_channels", ())), tuple(level.get("surrogate_channels", ()))),
            claim_counts=ClaimCountSpec(int(counts.get("total", 0)), int(counts.get("supported", 0)), int(counts.get("contested", 0)), int(counts.get("insufficient_evidence", 0)), int(counts.get("not_applicable", 0))),
            available_claim_types=tuple(str(value) for value in overview.get("available_claim_types", ())),
        ),
        story=tuple(StoryStageSpec(str(item["stage_id"]), str(item["title"]), str(item["evidence_status"]), str(item["effect"]), str(item["severity"]), tuple(item.get("facts", ())), tuple(item.get("claim_refs", ()))) for item in data.get("story", ())),
        data_profile=tuple(FeatureProfileSpec(str(item["feature"]), _number(item.get("object_value")), _number(item.get("reference_median")), tuple(item.get("reference_interval", (None, None))), tuple(item["subgroup_interval"]) if item.get("subgroup_interval") else None, _number(item.get("percentile")), str(item.get("anomaly_status", "unknown")), _number(item.get("contribution")), str(item.get("explanation", "")), tuple(item.get("claim_refs", ()))) for item in data.get("data_profile", ())),
        training_timeline=tuple(TrainingTimelineSpec(str(item["object_id"]), tuple(TrainingPointSpec(int(point["epoch"]), point.get("correct"), _number(point.get("confidence")), _number(point.get("loss")), _number(point.get("margin")), _number(point.get("prototype_distance")), _number(point.get("global_metric")), _number(point.get("subgroup_metric")), tuple(RuleActivationSpec(str(value["rule_id"]), float(value["activation"])) for value in point.get("rule_activations", ())), bool(point.get("forgetting"))) for point in item.get("points", ())), tuple(item.get("annotations", ())), tuple(item.get("claim_refs", ()))) for item in data.get("training_timeline", ())),
        knowledge_atlas=KnowledgeAtlasSpec(tuple(rules), concepts, int(atlas.get("source_rule_count", len(rules))), int(atlas.get("displayed_rule_count", len(rules))), int(atlas.get("primary_rule_count", 0))),
        decision_evidence=DecisionEvidenceSpec(tuple(claim_item(item) for item in decision.get("supports", ())), tuple(claim_item(item) for item in decision.get("contradicts", ())), tuple(claim_item(item) for item in decision.get("limitations", ()))),
        similar_cases=tuple(SimilarCaseSpec(str(item.get("query_object_id", "unknown")), str(item["reference_object_id"]), float(item["score"]), str(item["method"]), str(item["representation"]), tuple(item.get("matched_features", ())), tuple(item.get("different_features", ())), tuple(item.get("matched_regions", ())), _number(item.get("coverage_score")), str(item.get("reference_label", "")), str(item.get("reference_prediction", "")), bool(item.get("is_counterexample", False)), tuple((str(pair[0]), str(pair[1])) for pair in item.get("media_artifacts", ())), tuple(item.get("limitations", ())), tuple(item.get("claim_refs", ()))) for item in data.get("similar_cases", ())),
        counterfactuals=tuple(CounterfactualSpec(str(item.get("source_prediction", "")), str(item.get("target_prediction", "")), tuple(FeatureChangeSpec(str(value["feature"]), None if value.get("source_value") is None else str(value["source_value"]), str(value["target_value"])) for value in item.get("changed_features", ())), tuple(item.get("changed_rules", ())), _number(item.get("minimality")), _number(item.get("plausibility")), _number(item.get("expected_effect")), _number(item.get("observed_effect")), str(item.get("actionability", "unknown")), tuple(item.get("limitations", ())), tuple(item.get("claim_refs", ()))) for item in data.get("counterfactuals", ())),
        rule_ablations=tuple(RuleAblationMetricSpec(str(item["rule_id"]), str(item["metric"]), float(item["with_rule"]), float(item["without_rule"]), float(item["difference"]), str(item["scope"])) for item in data.get("rule_ablations", ())),
        provenance_nodes=tuple(ProvenanceNodeSpec(str(item["node_id"]), str(item["node_type"]), str(item["label"])) for item in data.get("provenance_nodes", ())),
        provenance_edges=tuple(ProvenanceEdgeSpec(str(item["source"]), str(item["target"]), str(item["relation"])) for item in data.get("provenance_edges", ())),
        audit=AuditSpec(bool(audit.get("graph_valid")), tuple(audit.get("graph_errors", ())), int(audit.get("node_count", 0)), int(audit.get("edge_count", 0)), int(audit.get("claim_count", 0)), tuple(audit.get("missing_evidence", ()))),
        schema_version="1.1",
    )


def _claim_refs(claims: Sequence[ExplanationClaim], claim_type: str, subject_id: str | None = None) -> tuple[str, ...]:
    return tuple(
        claim.claim_id
        for claim in claims
        if claim.claim_type == claim_type and (subject_id is None or claim.subject_id == subject_id)
    )


def _claim_item(claim: ExplanationClaim) -> ClaimVisualSpec:
    return ClaimVisualSpec(
        claim_id=claim.claim_id,
        statement=claim.short_statement,
        evidence_status=claim.evidence_status,
        effect=claim.effect,
        severity=claim.severity,
        strength=claim.strength,
        evidence_refs=tuple(claim.evidence_refs),
        limitations=tuple(claim.limitations),
        native=claim.native,
        surrogate=claim.surrogate,
    )


def _text(value: object) -> str:
    if isinstance(value, str):
        return value
    return json.dumps(value, ensure_ascii=False, sort_keys=True, default=str)


def _number(value: object) -> float | None:
    return float(value) if isinstance(value, (int, float)) else None


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
    level = dict(explanation_level or {})
    claim_types = {claim.claim_type for claim in claims}

    def stage(stage_id: str, title: str, types: set[str]) -> StoryStageSpec:
        selected = [claim for claim in claims if claim.claim_type in types]
        if not selected:
            evidence_status = "insufficient_evidence"
            effect = "unknown"
            severity = "warning"
        else:
            evidence_status = (
                "contested"
                if any(claim.evidence_status == "contested" for claim in selected)
                else "insufficient_evidence"
                if any(claim.evidence_status == "insufficient_evidence" for claim in selected)
                else "supported"
            )
            effect = "adverse" if any(claim.effect == "adverse" for claim in selected) else "mixed" if len({claim.effect for claim in selected}) > 1 else selected[0].effect
            severity = "critical" if any(claim.severity == "critical" for claim in selected) else "warning" if any(claim.severity == "warning" for claim in selected) else "info"
        return StoryStageSpec(
            stage_id=stage_id,
            title=title,
            evidence_status=evidence_status,
            effect=effect,
            severity=severity,
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
            subgroup = item.subgroup_profiles.get(feature)
            value = raw.get(feature)
            missing = item.missingness.get(feature, False)
            anomalous = feature in item.anomaly_labels
            refs = tuple(
                claim.claim_id
                for claim in claims
                if claim.subject_id == item.object_id and claim.claim_type in {"data_quality", "data_deviation", "data_error_status"}
            )
            profiles.append(
                FeatureProfileSpec(
                    feature=feature,
                    object_value=_number(value),
                    reference_median=_number(profile.get("median")),
                    reference_interval=(_number(profile.get("q05")), _number(profile.get("q95"))),
                    subgroup_interval=(_number(subgroup.get("q05")), _number(subgroup.get("q95"))) if subgroup else None,
                    percentile=_number(profile.get("percentile")),
                    anomaly_status="missing" if missing else "deviation_not_error" if anomalous else "within_reference",
                    contribution=_number(contributions.get(feature)),
                    explanation="Значение отсутствует." if missing else "Отклонение от общего референса не доказывает ошибку данных." if anomalous else "Значение находится в наблюдаемом референсном профиле.",
                    claim_refs=refs,
                )
            )

    timelines: list[TrainingTimelineSpec] = []
    for trace in evidence.training:
        points = tuple(
            TrainingPointSpec(
                epoch=int(metric.get("epoch", index)),
                correct=bool(metric["correct"]) if "correct" in metric else None,
                confidence=_number(metric.get("confidence")),
                loss=_number(metric.get("loss")),
                margin=_number(metric.get("margin")),
                prototype_distance=_number(metric.get("prototype_distance")),
                global_metric=_number(metric.get("global_metric")),
                subgroup_metric=_number(metric.get("subgroup_metric")),
                rule_activations=tuple(RuleActivationSpec(str(key), float(value)) for key, value in dict(metric.get("rule_activations", {})).items()),
                forgetting=int(metric.get("epoch", index)) in trace.forgetting_events,
            )
            for index, metric in enumerate(trace.epoch_metrics)
        )
        refs = _claim_refs(claims, "first_learned", trace.object_id) + _claim_refs(claims, "forgetting", trace.object_id)
        timelines.append(TrainingTimelineSpec(trace.object_id, points, tuple(claim.short_statement for claim in claims if claim.claim_id in refs), refs))

    rules = tuple(
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
            counterfactual_effect=RuleEffectSpec(
                train=_number(rule.counterfactual_effect.get("train")),
                validation=_number(rule.counterfactual_effect.get("validation")),
                test=_number(rule.counterfactual_effect.get("test")),
                subgroup_recall=_number(rule.counterfactual_effect.get("subgroup_recall")),
                critical_errors=_number(rule.counterfactual_effect.get("critical_errors")),
                calibration=_number(rule.counterfactual_effect.get("calibration")),
                additional=tuple(
                    MetricValueSpec(name, float(value))
                    for name, value in sorted(rule.counterfactual_effect.items())
                    if name not in {"train", "validation", "test", "subgroup_recall", "critical_errors", "calibration"}
                ),
            ),
            claim_refs=_claim_refs(claims, "model_rule", rule.rule_id),
        )
        for rule in evidence.rules
    )
    concepts = tuple(
        ClassConceptSpec(
            class_id=concept.class_id,
            class_name=concept.class_name,
            description=concept.human_description,
            primary_rules=tuple(concept.primary_rules),
            primary_rule_coverage=concept.primary_rule_coverage,
            uncovered_fraction=concept.uncovered_fraction,
            representative_objects=tuple(concept.representative_objects),
            boundary_objects=tuple(concept.boundary_objects),
            claim_refs=_claim_refs(claims, "class_concept", concept.class_id),
        )
        for concept in evidence.concepts
    )
    support_types = {"prediction", "model_rule", "class_concept", "similar_case"}
    contradict_types = {"data_deviation", "forgetting", "subgroup_averaging", "lost_rules", "diagnostic"}
    supports = tuple(_claim_item(claim) for claim in claims if claim.claim_type in support_types and claim.evidence_status == "supported" and claim.effect != "adverse")
    contradicts = tuple(_claim_item(claim) for claim in claims if claim.claim_type in contradict_types and claim.evidence_status != "insufficient_evidence")
    limitations = tuple(_claim_item(claim) for claim in claims if claim.evidence_status in {"contested", "insufficient_evidence"})

    similar_specs = tuple(
        SimilarCaseSpec(
            query_object_id=item.query_object_id,
            reference_object_id=item.reference_object_id,
            score=item.similarity_score,
            method=item.similarity_method,
            representation=item.compared_representation,
            matched_features=tuple(item.matched_features),
            different_features=tuple(item.different_features),
            matched_regions=tuple(item.matched_regions),
            coverage_score=item.coverage_score,
            reference_label=_text(item.reference_label),
            reference_prediction=_text(item.reference_prediction),
            is_counterexample=item.is_counterexample,
            media_artifacts=tuple(sorted((str(name), str(path)) for name, path in item.media_artifacts.items())),
            limitations=tuple(item.limitations),
            claim_refs=_claim_refs(claims, "similar_case", item.query_object_id),
        )
        for item in evidence.similar_cases
    )
    counterfactual_specs = tuple(
        CounterfactualSpec(
            source_prediction=_text(item.source_prediction),
            target_prediction=_text(item.target_prediction),
            changed_features=tuple(FeatureChangeSpec(str(name), None, _text(value)) for name, value in item.changed_features.items()),
            changed_rules=tuple(item.changed_rules),
            minimality=item.minimality,
            plausibility=item.plausibility,
            expected_effect=item.expected_effect,
            observed_effect=item.observed_effect,
            actionability=item.actionability,
            limitations=tuple(item.limitations),
            claim_refs=_claim_refs(claims, "counterfactual"),
        )
        for item in evidence.counterfactuals
    )
    ablations = tuple(
        RuleAblationMetricSpec(
            rule_id=rule.rule_id,
            metric=name,
            with_rule=float(value),
            without_rule=float(rule.ablation_without_rule[name]),
            difference=float(value - rule.ablation_without_rule[name]),
            scope="subgroup" if name.startswith("subgroup_") or name.startswith("critical_") else name,
        )
        for rule in evidence.rules
        for name, value in sorted(rule.ablation_baseline.items())
        if name in rule.ablation_without_rule
    )
    graph_errors = graph.validate_reachability()
    prediction_spec = PredictionVisualSpec(
        prediction=_text(prediction.get("predictions")),
        score=_number(prediction.get("score")),
        model_type=str(prediction.get("model_type", "unknown")),
        adapter_id=str(prediction.get("adapter_id", "unknown")),
    )
    level_spec = ExplanationLevelVisualSpec(
        level=str(level.get("level", "E0")),
        rationale=str(level.get("rationale", "")),
        available_channels=tuple(str(item) for item in level.get("available_channels", ())),
        missing_channels=tuple(str(item) for item in level.get("missing_channels", ())),
        native_channels=tuple(str(item) for item in level.get("native_channels", ())),
        surrogate_channels=tuple(str(item) for item in level.get("surrogate_channels", ())),
    )
    return ExplanationVisualSpec(
        overview=OverviewSpec(
            prediction=prediction_spec,
            action=action,
            explanation_level=level_spec,
            claim_counts=ClaimCountSpec(
                total=len(claims),
                supported=sum(claim.evidence_status == "supported" for claim in claims),
                contested=sum(claim.evidence_status == "contested" for claim in claims),
                insufficient_evidence=sum(claim.evidence_status == "insufficient_evidence" for claim in claims),
                not_applicable=sum(claim.evidence_status == "not_applicable" for claim in claims),
            ),
            available_claim_types=tuple(sorted(claim_types)),
        ),
        story=story,
        data_profile=tuple(profiles),
        training_timeline=tuple(timelines),
        knowledge_atlas=KnowledgeAtlasSpec(rules, concepts, len(evidence.rules), len(rules), sum(rule.is_primary for rule in evidence.rules)),
        decision_evidence=DecisionEvidenceSpec(supports, contradicts, limitations),
        similar_cases=similar_specs,
        counterfactuals=counterfactual_specs,
        rule_ablations=ablations,
        provenance_nodes=tuple(ProvenanceNodeSpec(node.node_id, node.node_type, node.label) for node in graph.nodes),
        provenance_edges=tuple(ProvenanceEdgeSpec(edge.source, edge.target, edge.relation) for edge in graph.edges),
        audit=AuditSpec(not graph_errors, graph_errors, len(graph.nodes), len(graph.edges), len(claims), tuple(graph.missing_evidence)),
    )
