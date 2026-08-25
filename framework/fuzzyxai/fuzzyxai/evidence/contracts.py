from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass, field
from typing import Any, Literal, cast

EvidenceStatus = Literal["supported", "contested", "insufficient_evidence", "not_applicable"]
EffectDirection = Literal["favorable", "adverse", "neutral", "mixed", "unknown"]
Severity = Literal["info", "warning", "critical"]
AudienceName = Literal["domain_user", "ml_engineer", "auditor", "researcher"]
DomainLanguageStatus = Literal["available", "insufficient_domain_language"]
ReasonEffectDirection = Literal["supports", "opposes", "mixed", "additional_support"]
ResultOrigin = Literal["measured", "controlled_fixture", "derived", "expert_defined"]
CounterfactualMode = Literal["sensitivity_analysis", "actionable_counterfactual"]
ReviewStatus = Literal["not_reviewed", "reviewed", "rejected"]


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
class TrainingCheckpointEvidence(EvidenceContract):
    """Measured state of one training run at one checkpoint."""

    run_id: str
    checkpoint_id: str
    epoch: int
    model_fingerprint: str
    train_metric: float
    validation_metric: float
    test_metric: float
    subgroup_metric: float | None
    object_predictions: Mapping[str, Any]
    object_confidences: Mapping[str, float]
    object_losses: Mapping[str, float]
    object_margins: Mapping[str, float]
    active_rules: Mapping[str, Sequence[str]]
    nearest_neighbors: Mapping[str, Sequence[str]]
    random_seed: int
    captured_at: str
    result_origin: ResultOrigin = "measured"

    def __post_init__(self) -> None:
        if self.result_origin != "measured":
            raise ValueError("training checkpoints must be measured")
        if self.epoch < 1:
            raise ValueError("checkpoint epoch must be positive")
        if not self.model_fingerprint:
            raise ValueError("checkpoint requires a model fingerprint")


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
    reference_rank: int | None = None
    reference_count: int | None = None


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
    mode: CounterfactualMode = "sensitivity_analysis"
    actionable: bool | None = None
    probability_before: float | None = None
    probability_after: float | None = None

    def __post_init__(self) -> None:
        if self.mode == "actionable_counterfactual" and self.actionable is not True:
            raise ValueError("actionable_counterfactual requires actionable=True")
        if self.mode == "actionable_counterfactual" and (self.minimality is None or self.plausibility is None):
            raise ValueError("actionable_counterfactual requires minimality and plausibility")


@dataclass(frozen=True)
class TextSpan(EvidenceContract):
    """One character range in raw text tied to a specific measured feature."""

    start: int
    end: int
    feature_name: str
    direction: Literal["supports", "contradicts"]
    weight: float

    def __post_init__(self) -> None:
        if self.start < 0 or self.end <= self.start:
            raise ValueError("text span requires 0 <= start < end")
        if self.direction not in {"supports", "contradicts"}:
            raise ValueError(f"unsupported text span direction: {self.direction}")
        if not math.isfinite(self.weight):
            raise ValueError("text span weight must be a finite number")


@dataclass(frozen=True)
class TextHighlightEvidence(EvidenceContract):
    """Raw text with feature-contribution spans located lexically, never inferred."""

    object_id: str
    raw_text: str
    spans: Sequence[TextSpan]
    unmapped_features: Sequence[str] = field(default_factory=tuple)
    suppressed_matches: Sequence[str] = field(default_factory=tuple)
    limitations: Sequence[str] = field(
        default_factory=lambda: ("span location is lexical substring matching, not semantic attribution",)
    )

    def __post_init__(self) -> None:
        length = len(self.raw_text)
        previous_end = -1
        for span in self.spans:
            if span.end > length:
                raise ValueError(f"span end {span.end} exceeds raw_text length {length}")
            if span.start < previous_end:
                raise ValueError("spans must be sorted by start offset and must not overlap")
            previous_end = span.end


@dataclass(frozen=True)
class ImageRegion(EvidenceContract):
    """One named region of an explained image, from a caller-supplied boolean mask.

    Region geometry (``bounding_box``, ``pixel_count``) is measured directly
    from the mask — never inferred, never a fabricated heatmap. ``direction``
    and ``contribution`` are only populated when the caller's contribution
    mapping has a matching entry for ``name``; otherwise "unknown"/``None``.
    """

    name: str
    pixel_count: int
    bounding_box: tuple[int, int, int, int]  # (row_min, row_max, col_min, col_max), inclusive
    direction: Literal["supports", "contradicts", "unknown"]
    contribution: float | None

    def __post_init__(self) -> None:
        if self.pixel_count <= 0:
            raise ValueError("image region requires at least one pixel")
        if self.direction not in {"supports", "contradicts", "unknown"}:
            raise ValueError(f"unsupported image region direction: {self.direction}")


@dataclass(frozen=True)
class ImageRepresentationEvidence(EvidenceContract):
    """Raw image dimensions/artifact plus explicitly-supplied region masks.

    ``image_png_base64`` carries the actual image content, analogous to
    ``TextHighlightEvidence.raw_text`` — kept in-memory for rendering, and
    redacted (like raw text) from JSON exports by default (``include_raw``).
    ``artifact_sha256`` is retained even when the image content is redacted,
    as an integrity reference without the content itself.
    """

    object_id: str
    width: int
    height: int
    channels: int
    artifact_sha256: str
    image_png_base64: str
    regions: Sequence[ImageRegion] = field(default_factory=tuple)
    limitations: Sequence[str] = field(default_factory=tuple)

    def __post_init__(self) -> None:
        if self.width <= 0 or self.height <= 0:
            raise ValueError("image representation requires positive width and height")
        if self.channels <= 0:
            raise ValueError("image representation requires at least one channel")


@dataclass(frozen=True)
class FuzzyTermMembership(EvidenceContract):
    """One antecedent term's measured membership degree for this object.

    ``membership_degree`` is a real [0, 1] value the rule/fuzzy model itself
    computed (e.g. a Gaussian/triangular membership function evaluated at
    the object's feature value) — never inferred or guessed by FuzzyXAI.
    """

    feature: str
    term: str
    membership_degree: float
    feature_value: float | None = None

    def __post_init__(self) -> None:
        if not self.feature.strip() or not self.term.strip():
            raise ValueError("fuzzy term requires a feature name and a term label")
        if not 0.0 <= self.membership_degree <= 1.0:
            raise ValueError("membership degree must be between 0 and 1")


@dataclass(frozen=True)
class FuzzyRuleActivation(EvidenceContract):
    """One rule's real activation for this object — not a re-labeled linear contribution.

    Any rule/fuzzy model can supply this evidence through the generic
    ``activated_rules`` channel on ``extract_local_evidence`` (a plain list
    of dicts with this shape) — the contract is not tied to any specific
    ANFIS library. ``activation_strength`` is typically the T-norm
    (e.g. product or min) of the rule's own term memberships, computed by
    the model, not derived here.
    """

    object_id: str
    rule_id: str
    terms: Sequence[FuzzyTermMembership]
    activation_strength: float
    conclusion: str
    limitations: Sequence[str] = field(default_factory=tuple)

    def __post_init__(self) -> None:
        if not self.rule_id.strip():
            raise ValueError("fuzzy rule activation requires a rule_id")
        if not self.terms:
            raise ValueError("fuzzy rule activation requires at least one antecedent term")
        if not 0.0 <= self.activation_strength <= 1.0:
            raise ValueError("activation strength must be between 0 and 1")
        if not str(self.conclusion).strip():
            raise ValueError("fuzzy rule activation requires a conclusion")


@dataclass(frozen=True)
class AtomicClaim(EvidenceContract):
    """One verbalizer-facing statement extracted from an already-verified HumanExplanation.

    This is the *only* thing a verbalization backend ever sees — never the raw
    prediction, evidence graph, or ExplanationClaim internals. ``allowed_numbers``
    and ``allowed_entities`` are the grounding guard's source of truth: any
    number or name a backend introduces that isn't in this closure is rejected.
    """

    claim_id: str
    kind: Literal["decision", "reason", "concern", "reliability", "action"]
    subject: str
    canonical_text: str
    allowed_numbers: Sequence[str] = field(default_factory=tuple)
    allowed_entities: Sequence[str] = field(default_factory=tuple)
    direction: str = "neutral"
    source_claim_ids: Sequence[str] = field(default_factory=tuple)


@dataclass(frozen=True)
class RuleAblationEvidence(EvidenceContract):
    """Before/after metrics measured by suppressing one rule or concept."""

    run_id: str
    rule_id: str
    model_fingerprint: str
    native: bool
    surrogate: bool
    fidelity: float | None
    train_metrics_with_rule: Mapping[str, float]
    validation_metrics_with_rule: Mapping[str, float]
    test_metrics_with_rule: Mapping[str, float]
    train_metrics_without_rule: Mapping[str, float]
    validation_metrics_without_rule: Mapping[str, float]
    test_metrics_without_rule: Mapping[str, float]
    subgroup_metrics_with_rule: Mapping[str, float]
    subgroup_metrics_without_rule: Mapping[str, float]
    critical_errors_with_rule: int
    critical_errors_without_rule: int
    target_prediction_with_rule: Any
    target_prediction_without_rule: Any
    limitations: Sequence[str]
    result_origin: ResultOrigin = "measured"

    def __post_init__(self) -> None:
        if self.surrogate and self.fidelity is None:
            raise ValueError("surrogate ablation requires fidelity")
        if self.native and self.surrogate:
            raise ValueError("ablation cannot be both native and surrogate")


@dataclass(frozen=True)
class DomainFeatureLanguage(EvidenceContract):
    label: str
    meaning: str
    unit: str | None = None
    high_text: str | None = None
    low_text: str | None = None
    positive_effect_text: str | None = None
    negative_effect_text: str | None = None
    expected_direction: Literal["increases_target", "decreases_target", "non_monotonic", "unknown"] = "unknown"
    expert_review_status: ReviewStatus = "not_reviewed"
    reviewer_role: str | None = None
    reviewed_at: str | None = None

    def __post_init__(self) -> None:
        if self.expert_review_status == "reviewed" and not (self.reviewer_role and self.reviewed_at):
            raise ValueError("reviewed domain language requires reviewer role and review timestamp")


@dataclass(frozen=True)
class DomainLanguageValidation(EvidenceContract):
    version: str
    language_hash: str
    status: Literal["pass", "insufficient_domain_language", "rejected"]
    checked_features: int
    errors: Sequence[str]
    warnings: Sequence[str]
    expert_review_required: bool


@dataclass(frozen=True)
class ComparisonStatement(EvidenceContract):
    sample_size: int
    reference_label: str
    representation: str
    percentile: float | None
    rank: int | None
    wording_policy: Literal["small_sample_rank", "medium_sample_tail", "large_sample_percentile", "insufficient_evidence"]
    text: str
    limitations: Sequence[str] = field(default_factory=tuple)


@dataclass(frozen=True)
class SimilarCaseExplanation(EvidenceContract):
    object_id: str
    role: Literal["support", "counterexample"]
    similarity_score: float
    similarity_method: str
    representation: str
    matched_features: Sequence[str]
    differing_features: Sequence[str]
    limitations: Sequence[str]


@dataclass(frozen=True)
class CounterfactualExplanation(EvidenceContract):
    mode: CounterfactualMode
    feature: str
    original_value: float
    changed_value: float
    unit: str | None
    prediction_before: Any
    prediction_after: Any
    probability_before: float | None
    probability_after: float | None
    minimality: float | None
    plausibility: float | None
    actionable: bool | None
    observed_effect: bool
    limitations: Sequence[str]

    def __post_init__(self) -> None:
        if self.mode == "actionable_counterfactual" and self.actionable is not True:
            raise ValueError("actionable counterfactual explanation requires domain-validated actionability")


@dataclass(frozen=True)
class ExplanationNode(EvidenceContract):
    node_id: str
    node_type: str
    label: str
    payload: Mapping[str, Any]
    evidence_refs: Sequence[str] = field(default_factory=tuple)

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> ExplanationNode:
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
    def from_dict(cls, payload: Mapping[str, Any]) -> ExplanationEdge:
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
    def from_dict(cls, payload: Mapping[str, Any]) -> ExplanationClaim:
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
    def from_dict(cls, payload: Mapping[str, Any]) -> ExplanationGraph:
        return cls(
            nodes=tuple(ExplanationNode.from_dict(item) for item in payload.get("nodes", ())),
            edges=tuple(ExplanationEdge.from_dict(item) for item in payload.get("edges", ())),
            claims=tuple(ExplanationClaim.from_dict(item) for item in payload.get("claims", ())),
            missing_evidence=tuple(str(item) for item in payload.get("missing_evidence", ())),
            schema_version=str(payload.get("schema_version", "2.0")),
        )

    def _node_map(self) -> dict[str, ExplanationNode]:
        return {node.node_id: node for node in self.nodes}

    def trace_claim(self, claim_id: str) -> ExplanationGraph:
        normalized = claim_id if claim_id.startswith("C-") else claim_id.replace("C", "C-", 1)
        return self._reachable_subgraph({f"claim:{normalized}"}, reverse=True)

    def trace_action(self) -> ExplanationGraph:
        return self._reachable_subgraph({"action"}, reverse=True)

    def subgraph(self, *, subject_id: str) -> ExplanationGraph:
        seeds = {
            node.node_id
            for node in self.nodes
            if str(node.payload.get("object_id", node.payload.get("subject_id", ""))) == subject_id
            or subject_id in node.node_id
        }
        return self._reachable_subgraph(seeds, reverse=False)

    def _reachable_subgraph(self, seeds: set[str], *, reverse: bool) -> ExplanationGraph:
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
    domain_language_status: DomainLanguageStatus

    def __post_init__(self) -> None:
        super().__post_init__()
        if self.domain_language_status not in {"available", "insufficient_domain_language"}:
            raise ValueError("unsupported domain-language status")


@dataclass(frozen=True)
class ReasonStatement(HumanStatement):
    subject_label: str
    effect_direction: ReasonEffectDirection
    comparison_text: str

    def __post_init__(self) -> None:
        super().__post_init__()
        if not self.subject_label.strip():
            raise ValueError("human reason requires a concrete subject label")
        if self.effect_direction not in {"supports", "opposes", "mixed", "additional_support"}:
            raise ValueError("unsupported human-reason effect direction")
        if not self.comparison_text.strip():
            raise ValueError("human reason requires an explicit comparison")


@dataclass(frozen=True)
class ConcernStatement(HumanStatement):
    pass


@dataclass(frozen=True)
class ReliabilityStatement(HumanStatement):
    supported_by: Sequence[str]
    limited_by: Sequence[str]
    missing_evidence: Sequence[str]
    conclusion: str

    def __post_init__(self) -> None:
        super().__post_init__()
        if not self.conclusion.strip():
            raise ValueError("reliability statement requires a conclusion")
        if not (self.supported_by or self.limited_by or self.missing_evidence):
            raise ValueError("reliability statement requires concrete support, limitation, or missing evidence")


@dataclass(frozen=True)
class ActionStatement(HumanStatement):
    action: str


@dataclass(frozen=True)
class ChangeStatement(HumanStatement):
    feature: str
    original_value: Any
    changed_value: Any
    direction: str
    prediction_before: Any
    prediction_after: Any
    observed_effect: float | None
    plausibility: float | None
    actionability: str

    def __post_init__(self) -> None:
        super().__post_init__()
        if not self.feature.strip() or self.original_value is None or self.changed_value is None:
            raise ValueError("human counterfactual requires feature, original value, and changed value")
        if not self.direction.strip() or self.prediction_before is None or self.prediction_after is None:
            raise ValueError("human counterfactual requires direction and predictions before/after")
        if not self.actionability.strip():
            raise ValueError("human counterfactual requires an actionability statement")


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
            # Prefix with the statement's own (already human-readable, not a
            # technical identifier) title when a section holds more than one
            # item — otherwise several distinct reasons whose explanation
            # text happens to share the same template read as duplicates.
            if len(statements) > 1:
                lines.extend(f"**{statement.title}.** {statement.explanation}" for statement in statements)
            else:
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
    def from_dict(cls, payload: Mapping[str, Any], *, graph: ExplanationGraph | None = None) -> HumanExplanation:
        def statement(kind: type[HumanStatement], value: Mapping[str, Any]) -> HumanStatement:
            title = str(value["title"])
            explanation = str(value["explanation"])
            claim_refs = tuple(str(item) for item in value.get("claim_refs", ()))
            evidence_refs = tuple(str(item) for item in value.get("evidence_refs", ()))
            if kind is DecisionStatement:
                return DecisionStatement(
                    title,
                    explanation,
                    claim_refs,
                    evidence_refs,
                    cast(DomainLanguageStatus, str(value.get("domain_language_status", "available"))),
                )
            if kind is ReasonStatement:
                return ReasonStatement(
                    title,
                    explanation,
                    claim_refs,
                    evidence_refs,
                    str(value.get("subject_label", title)),
                    cast(ReasonEffectDirection, str(value.get("effect_direction", "supports"))),
                    str(value.get("comparison_text", explanation)),
                )
            if kind is ReliabilityStatement:
                return ReliabilityStatement(
                    title,
                    explanation,
                    claim_refs,
                    evidence_refs,
                    tuple(str(item) for item in value.get("supported_by", ())),
                    tuple(str(item) for item in value.get("limited_by", ())),
                    tuple(str(item) for item in value.get("missing_evidence", ())),
                    str(value.get("conclusion", explanation)),
                )
            if kind is ActionStatement:
                return ActionStatement(title, explanation, claim_refs, evidence_refs, str(value.get("action", "review")))
            if kind is ChangeStatement:
                return ChangeStatement(
                    title,
                    explanation,
                    claim_refs,
                    evidence_refs,
                    str(value.get("feature", "")),
                    value.get("original_value"),
                    value.get("changed_value"),
                    str(value.get("direction", "")),
                    value.get("prediction_before"),
                    value.get("prediction_after"),
                    float(value["observed_effect"]) if value.get("observed_effect") is not None else None,
                    float(value["plausibility"]) if value.get("plausibility") is not None else None,
                    str(value.get("actionability", "")),
                )
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
    text_highlights: Sequence[TextHighlightEvidence] = field(default_factory=tuple)
    image_representations: Sequence[ImageRepresentationEvidence] = field(default_factory=tuple)
    fuzzy_rule_activations: Sequence[FuzzyRuleActivation] = field(default_factory=tuple)
    missing: Sequence[str] = field(default_factory=tuple)
