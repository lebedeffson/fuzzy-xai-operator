from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Literal, Mapping, Sequence, cast


EvidenceStatus = Literal["supported", "contested", "insufficient_evidence", "not_applicable"]
EffectDirection = Literal["favorable", "adverse", "neutral", "mixed", "unknown"]
Severity = Literal["info", "warning", "critical"]
AudienceName = Literal["domain_user", "ml_engineer", "auditor", "researcher"]


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
        payload = asdict(self)  # type: ignore[call-overload]
        return cast(dict[str, Any], _jsonable(payload))

    def __getitem__(self, key: str) -> Any:
        return self.to_dict()[key]

    def get(self, key: str, default: Any = None) -> Any:
        return self.to_dict().get(key, default)


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
    reference_profiles: Mapping[str, Mapping[str, float | None]] = field(default_factory=dict)
    subgroup_profiles: Mapping[str, Mapping[str, float | None]] = field(default_factory=dict)


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
    ablation_baseline: Mapping[str, float] = field(default_factory=dict)
    ablation_without_rule: Mapping[str, float] = field(default_factory=dict)


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
    is_counterexample: bool = False
    media_artifacts: Mapping[str, str] = field(default_factory=dict)


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

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "ExplanationNode":
        return cls(
            node_id=str(payload["node_id"]),
            node_type=str(payload["node_type"]),
            label=str(payload.get("label", "")),
            payload=dict(payload.get("payload", {})),
            evidence_refs=tuple(str(item) for item in payload.get("evidence_refs", ())),
        )


@dataclass(frozen=True)
class ExplanationEdge(EvidenceContract):
    source: str
    target: str
    relation: str
    evidence_refs: Sequence[str] = field(default_factory=tuple)

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "ExplanationEdge":
        return cls(
            source=str(payload["source"]),
            target=str(payload["target"]),
            relation=str(payload["relation"]),
            evidence_refs=tuple(str(item) for item in payload.get("evidence_refs", ())),
        )


@dataclass(frozen=True)
class ExplanationClaim(EvidenceContract):
    """One auditable statement derived from explicit evidence references."""

    claim_id: str
    claim_type: str
    scope: str
    subject_id: str
    statement: str
    short_statement: str
    evidence_status: EvidenceStatus
    effect: EffectDirection
    severity: Severity
    strength: float | None
    evidence_refs: Sequence[str]
    confidence_interval: tuple[float, float] | None = None
    counter_evidence_refs: Sequence[str] = field(default_factory=tuple)
    limitations: Sequence[str] = field(default_factory=tuple)
    applicability: str | None = None
    metric_name: str | None = None
    metric_value: float | None = None
    metric_unit: str | None = None
    comparison_baseline: str | None = None
    native: bool | None = None
    surrogate: bool | None = None

    @property
    def status(self) -> EvidenceStatus:
        """Compatibility alias for the pre-1.1 claim contract."""

        return self.evidence_status

    def to_dict(self) -> dict[str, Any]:
        payload = super().to_dict()
        payload["status"] = self.evidence_status
        return payload

    def __post_init__(self) -> None:
        if self.evidence_status not in {"supported", "contested", "insufficient_evidence", "not_applicable"}:
            raise ValueError(f"unsupported claim status: {self.evidence_status}")
        if self.effect not in {"favorable", "adverse", "neutral", "mixed", "unknown"}:
            raise ValueError(f"unsupported claim effect: {self.effect}")
        if self.severity not in {"info", "warning", "critical"}:
            raise ValueError(f"unsupported claim severity: {self.severity}")
        if self.evidence_status == "supported" and not self.evidence_refs:
            raise ValueError("supported claim requires at least one evidence reference")
        if self.evidence_status == "contested" and not (self.evidence_refs or self.counter_evidence_refs):
            raise ValueError("contested claim requires evidence or counter-evidence")
        if self.evidence_status == "insufficient_evidence" and not self.limitations:
            raise ValueError("insufficient_evidence claim requires a missing-channel limitation")
        if self.strength is not None and not 0.0 <= self.strength <= 1.0:
            raise ValueError("claim strength must be between 0 and 1")
        if self.confidence_interval is not None:
            low, high = self.confidence_interval
            if low > high:
                raise ValueError("confidence interval lower bound exceeds upper bound")
        if self.metric_value is not None and not self.metric_name:
            raise ValueError("metric_value requires metric_name")
        if self.surrogate and not any("fidelity" in item.lower() for item in self.limitations):
            raise ValueError("surrogate claim requires a fidelity limitation")
        if self.scope == "medical" and self.applicability != "research_only":
            raise ValueError("medical claims require applicability='research_only'")

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "ExplanationClaim":
        interval = payload.get("confidence_interval")
        return cls(
            claim_id=str(payload["claim_id"]),
            claim_type=str(payload["claim_type"]),
            scope=str(payload["scope"]),
            subject_id=str(payload["subject_id"]),
            statement=str(payload["statement"]),
            short_statement=str(payload["short_statement"]),
            evidence_status=cast(EvidenceStatus, str(payload.get("evidence_status", payload.get("status", "insufficient_evidence")))),
            effect=cast(EffectDirection, str(payload.get("effect", "unknown"))),
            severity=cast(Severity, str(payload.get("severity", "warning"))),
            strength=None if payload.get("strength") is None else float(payload["strength"]),
            evidence_refs=tuple(str(item) for item in payload.get("evidence_refs", ())),
            confidence_interval=(float(interval[0]), float(interval[1])) if isinstance(interval, (list, tuple)) and len(interval) == 2 else None,
            counter_evidence_refs=tuple(str(item) for item in payload.get("counter_evidence_refs", ())),
            limitations=tuple(str(item) for item in payload.get("limitations", ())),
            applicability=None if payload.get("applicability") is None else str(payload["applicability"]),
            metric_name=None if payload.get("metric_name") is None else str(payload["metric_name"]),
            metric_value=None if payload.get("metric_value") is None else float(payload["metric_value"]),
            metric_unit=None if payload.get("metric_unit") is None else str(payload["metric_unit"]),
            comparison_baseline=None if payload.get("comparison_baseline") is None else str(payload["comparison_baseline"]),
            native=None if payload.get("native") is None else bool(payload["native"]),
            surrogate=None if payload.get("surrogate") is None else bool(payload["surrogate"]),
        )


@dataclass(frozen=True)
class ExplanationLevel(EvidenceContract):
    """Honest disclosure of explanation depth and channel provenance."""

    level: str
    available_channels: Sequence[str]
    missing_channels: Sequence[str]
    native_channels: Sequence[str]
    surrogate_channels: Sequence[str]
    rationale: str

    def __post_init__(self) -> None:
        if self.level not in {"E0", "E1", "E2", "E3", "E4", "E5"}:
            raise ValueError(f"unsupported explanation level: {self.level}")


@dataclass(frozen=True)
class ExplanationGraph(EvidenceContract):
    nodes: Sequence[ExplanationNode]
    edges: Sequence[ExplanationEdge]
    claims: Sequence[ExplanationClaim] = field(default_factory=tuple)
    missing_evidence: Sequence[str] = field(default_factory=tuple)
    schema_version: str = "2.0"

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "ExplanationGraph":
        return cls(
            nodes=tuple(ExplanationNode.from_dict(item) for item in payload.get("nodes", ())),
            edges=tuple(ExplanationEdge.from_dict(item) for item in payload.get("edges", ())),
            claims=tuple(ExplanationClaim.from_dict(item) for item in payload.get("claims", ())),
            missing_evidence=tuple(str(item) for item in payload.get("missing_evidence", ())),
            schema_version=str(payload.get("schema_version", "2.0")),
        )

    def _node_map(self) -> dict[str, ExplanationNode]:
        return {node.node_id: node for node in self.nodes}

    def trace_claim(self, claim_id: str) -> "ExplanationGraph":
        normalized = claim_id if claim_id.startswith("C-") else claim_id.replace("C", "C-", 1)
        return self._reachable_subgraph({f"claim:{normalized}"}, reverse=True)

    def trace_action(self) -> "ExplanationGraph":
        return self._reachable_subgraph({"action"}, reverse=True)

    def subgraph(self, *, subject_id: str) -> "ExplanationGraph":
        seeds = {
            node.node_id
            for node in self.nodes
            if str(node.payload.get("object_id", node.payload.get("subject_id", ""))) == subject_id
            or subject_id in node.node_id
        }
        return self._reachable_subgraph(seeds, reverse=False)

    def _reachable_subgraph(self, seeds: set[str], *, reverse: bool) -> "ExplanationGraph":
        selected = set(seeds)
        changed = True
        while changed:
            changed = False
            for edge in self.edges:
                anchor = edge.target if reverse else edge.source
                related = edge.source if reverse else edge.target
                if anchor in selected and related not in selected:
                    selected.add(related)
                    changed = True
        return ExplanationGraph(
            nodes=tuple(node for node in self.nodes if node.node_id in selected),
            edges=tuple(edge for edge in self.edges if edge.source in selected and edge.target in selected),
            claims=tuple(claim for claim in self.claims if f"claim:{claim.claim_id}" in selected),
            missing_evidence=self.missing_evidence,
            schema_version=self.schema_version,
        )

    def validate_reachability(self) -> tuple[str, ...]:
        nodes = self._node_map()
        errors = [
            f"dangling edge {edge.source}->{edge.target}"
            for edge in self.edges
            if edge.source not in nodes or edge.target not in nodes
        ]
        incoming: dict[str, list[ExplanationEdge]] = {}
        for edge in self.edges:
            incoming.setdefault(edge.target, []).append(edge)
        for claim in self.claims:
            if claim.evidence_status == "supported" and not any(
                edge.relation in {"supports_claim", "derived_from", "observed_during", "lost_during", "changed_by"}
                for edge in incoming.get(f"claim:{claim.claim_id}", [])
            ):
                errors.append(f"supported claim {claim.claim_id} is not reachable from evidence")
        action_sources = {edge.source for edge in incoming.get("action", [])}
        if "prediction" not in action_sources:
            errors.append("action is not reachable from prediction")
        if not any(source.startswith(("claim:", "diagnostic:")) for source in action_sources):
            errors.append("action is not constrained by a claim or diagnostic")
        return tuple(errors)


@dataclass(frozen=True)
class AudienceProfile(EvidenceContract):
    name: AudienceName
    max_reasons: int
    max_concerns: int
    max_changes: int
    show_technical_identifiers: bool
    include_metrics: bool


@dataclass(frozen=True)
class HumanStatement(EvidenceContract):
    title: str
    explanation: str
    claim_refs: Sequence[str]
    evidence_refs: Sequence[str]

    def __post_init__(self) -> None:
        if not self.title.strip() or not self.explanation.strip():
            raise ValueError("human statement requires title and explanation")
        if not self.claim_refs:
            raise ValueError("human statement requires at least one claim reference")
        if not self.evidence_refs:
            raise ValueError("human statement requires at least one evidence reference")


@dataclass(frozen=True)
class DecisionStatement(HumanStatement):
    pass


@dataclass(frozen=True)
class ReasonStatement(HumanStatement):
    pass


@dataclass(frozen=True)
class ConcernStatement(HumanStatement):
    pass


@dataclass(frozen=True)
class ReliabilityStatement(HumanStatement):
    pass


@dataclass(frozen=True)
class ActionStatement(HumanStatement):
    action: str


@dataclass(frozen=True)
class ChangeStatement(HumanStatement):
    pass


@dataclass(frozen=True)
class ExplanationDetails(EvidenceContract):
    supports: Sequence[ReasonStatement] = field(default_factory=tuple)
    contradicts: Sequence[ConcernStatement] = field(default_factory=tuple)
    limitations: Sequence[ConcernStatement] = field(default_factory=tuple)
    training: Sequence[HumanStatement] = field(default_factory=tuple)
    similar_cases: Sequence[HumanStatement] = field(default_factory=tuple)
    technical_metrics: Sequence[HumanStatement] = field(default_factory=tuple)


@dataclass(frozen=True)
class HumanExplanation(EvidenceContract):
    audience: AudienceName
    language: str
    decision: DecisionStatement
    main_reasons: Sequence[ReasonStatement]
    concerns: Sequence[ConcernStatement]
    reliability: ReliabilityStatement
    recommended_action: ActionStatement
    what_would_change_result: Sequence[ChangeStatement]
    details: ExplanationDetails
    technical_trace: ExplanationGraph

    @property
    def fragments(self) -> tuple[HumanStatement, ...]:
        return (
            self.decision,
            *self.main_reasons,
            *self.concerns,
            self.reliability,
            self.recommended_action,
            *self.what_would_change_result,
        )

    @property
    def user_text(self) -> str:
        sections = [
            ("Решение", (self.decision,)),
            ("Почему", self.main_reasons),
            ("Что вызывает сомнение", self.concerns),
            ("Насколько можно доверять", (self.reliability,)),
            ("Что делать", (self.recommended_action,)),
            ("Что изменит результат", self.what_would_change_result),
        ]
        lines: list[str] = []
        for heading, statements in sections:
            if not statements:
                continue
            lines.append(f"## {heading}")
            lines.extend(statement.explanation for statement in statements)
            lines.append("")
        return "\n".join(lines).strip() + "\n"

    def to_dict(self, *, include_technical_trace: bool = True) -> dict[str, Any]:
        payload = super().to_dict()
        if not include_technical_trace:
            payload.pop("technical_trace", None)
        # One-cycle compatibility for existing dashboard and export consumers.
        payload.update(
            {
                "level": {"domain_user": "user", "ml_engineer": "expert", "auditor": "audit", "researcher": "expert"}[self.audience],
                "summary": self.user_text,
                "model_observed": [self.decision.explanation],
                "lost_or_averaged": [item.explanation for item in self.concerns],
                "similar_cases": [item.explanation for item in self.details.similar_cases],
                "decision_changes": [item.explanation for item in self.what_would_change_result],
                "trust": [self.reliability.explanation, self.recommended_action.explanation],
                "limitations": [item.explanation for item in self.details.limitations],
                "evidence_trace": [] if self.audience == "domain_user" else [node.node_id for node in self.technical_trace.nodes],
                "claim_refs": {
                    "decision": list(self.decision.claim_refs),
                    "main_reasons": [ref for item in self.main_reasons for ref in item.claim_refs],
                    "concerns": [ref for item in self.concerns for ref in item.claim_refs],
                    "reliability": list(self.reliability.claim_refs),
                    "recommended_action": list(self.recommended_action.claim_refs),
                    "changes": [ref for item in self.what_would_change_result for ref in item.claim_refs],
                },
            }
        )
        return payload

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any], *, graph: ExplanationGraph | None = None) -> "HumanExplanation":
        def statement(kind: type[HumanStatement], value: Mapping[str, Any]) -> HumanStatement:
            title = str(value["title"])
            explanation = str(value["explanation"])
            claim_refs = tuple(str(item) for item in value.get("claim_refs", ()))
            evidence_refs = tuple(str(item) for item in value.get("evidence_refs", ()))
            if kind is ActionStatement:
                return ActionStatement(title, explanation, claim_refs, evidence_refs, str(value.get("action", "review")))
            return kind(title, explanation, claim_refs, evidence_refs)

        details_payload = payload.get("details", {})
        trace_payload = payload.get("technical_trace", {})
        trace = graph or (ExplanationGraph.from_dict(trace_payload) if isinstance(trace_payload, Mapping) and trace_payload else ExplanationGraph((), (), ()))
        return cls(
            audience=cast(AudienceName, str(payload.get("audience", "domain_user"))),
            language=str(payload.get("language", "ru")),
            decision=cast(DecisionStatement, statement(DecisionStatement, cast(Mapping[str, Any], payload["decision"]))),
            main_reasons=tuple(cast(ReasonStatement, statement(ReasonStatement, item)) for item in payload.get("main_reasons", ())),
            concerns=tuple(cast(ConcernStatement, statement(ConcernStatement, item)) for item in payload.get("concerns", ())),
            reliability=cast(ReliabilityStatement, statement(ReliabilityStatement, cast(Mapping[str, Any], payload["reliability"]))),
            recommended_action=cast(ActionStatement, statement(ActionStatement, cast(Mapping[str, Any], payload["recommended_action"]))),
            what_would_change_result=tuple(cast(ChangeStatement, statement(ChangeStatement, item)) for item in payload.get("what_would_change_result", ())),
            details=ExplanationDetails(
                supports=tuple(cast(ReasonStatement, statement(ReasonStatement, item)) for item in details_payload.get("supports", ())),
                contradicts=tuple(cast(ConcernStatement, statement(ConcernStatement, item)) for item in details_payload.get("contradicts", ())),
                limitations=tuple(cast(ConcernStatement, statement(ConcernStatement, item)) for item in details_payload.get("limitations", ())),
                training=tuple(statement(HumanStatement, item) for item in details_payload.get("training", ())),
                similar_cases=tuple(statement(HumanStatement, item) for item in details_payload.get("similar_cases", ())),
                technical_metrics=tuple(statement(HumanStatement, item) for item in details_payload.get("technical_metrics", ())),
            ),
            technical_trace=trace,
        )


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
