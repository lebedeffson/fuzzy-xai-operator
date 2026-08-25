from __future__ import annotations

import json
import shutil
import zipfile
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from hashlib import sha256
from html import escape
from pathlib import Path
from typing import TYPE_CHECKING, Any, cast

if TYPE_CHECKING:
    from fuzzyxai.verbalization import VerbalizationBackend, VerbalizationResult

from fuzzyxai.adapters.base import BaseAdapter
from fuzzyxai.adapters.contracts_v2 import AdapterResolutionReport, ExplanationContext, ModelCapabilities, TaskType
from fuzzyxai.adapters.model import ModelAdapter, ModelPrediction
from fuzzyxai.adapters.model_registry import MODEL_ADAPTER_REGISTRY, resolve_model_adapter_v2
from fuzzyxai.adapters.model_v2 import ModelAdapterV2
from fuzzyxai.core.explain_plan import ExplainPlan
from fuzzyxai.core.route import build_route
from fuzzyxai.core.types import AdaptedInput, OperatorRoute
from fuzzyxai.diagnostics.contracts import (
    BatchDiagnosticReport,
    DiagnosticReport,
    RepairExecutionContext,
)
from fuzzyxai.evidence import (
    ExplanationClaim,
    ExplanationEdge,
    ExplanationEvidence,
    ExplanationGraph,
    ExplanationNode,
    HumanExplanation,
    TrainingRunAnalysis,
    build_class_concepts,
    build_explanation_claims,
    build_explanation_graph,
    build_object_trace,
    collect_data_evidence,
    collect_fuzzy_rule_activations,
    compose_human_explanation,
    detect_subgroup_averaging,
    determine_explanation_level,
    evaluate_explanation_quality,
    explanation_to_text,
    extract_rules,
    find_image_regions,
    find_similar_tabular_cases,
    find_tabular_counterfactuals,
    find_text_highlight_spans,
    is_image_like,
)
from fuzzyxai.explanation_quality import ExplanationQualityReport, build_quality_report
from fuzzyxai.operators import AlignmentInput, ReductionInput, RiskInput
from fuzzyxai.operators import compute_alignment as compute_operator_alignment
from fuzzyxai.operators import compute_reduction as compute_operator_reduction
from fuzzyxai.operators import observe_risk as observe_operator_risk
from fuzzyxai.planner import ExplanationPlanner
from fuzzyxai.proof.trace import build_proof_trace
from fuzzyxai.proof.verifier import VerificationResult, verify_proof_trace
from fuzzyxai.visualization.operator_dashboard import render_dashboard
from fuzzyxai.visualization.route_artifacts import save_proof_trace_json, save_route_json
from fuzzyxai.visualization.spec import build_visual_spec
from fuzzyxai.visualization.traceability import write_traceability_artifacts
from fuzzyxai.visualization.view_model import _RAW_IMAGE_REDACTED, _RAW_TEXT_REDACTED, ExplanationViewModel


def _as_rows(values: Any) -> list[list[Any]]:
    if hasattr(values, "to_numpy"):
        values = values.to_numpy()
    if hasattr(values, "tolist"):
        values = values.tolist()
    if not isinstance(values, (list, tuple)):
        raise TypeError("explanation inputs must be a row or a sequence of rows")
    values = list(values)
    if not values:
        raise ValueError("explanation inputs cannot be empty")
    if not isinstance(values[0], (list, tuple)):
        return [list(values)]
    return [list(row) for row in values]


def _default_feature_names(rows: list[list[Any]]) -> list[str]:
    return [f"feature_{index}" for index in range(len(rows[0]))]


def _payload_sha256(value: Any) -> str:
    canonical = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), default=str)
    return sha256(canonical.encode("utf-8")).hexdigest()


def _with_feature_names(internal: Mapping[str, Any], names: list[str]) -> dict[str, Any]:
    result = dict(internal)
    contributions = result.get("contributions")
    if isinstance(contributions, Mapping):
        keys = list(contributions)
        if len(keys) == len(names) and keys == [f"feature_{index}" for index in range(len(keys))]:
            result["contributions"] = {name: contributions[key] for name, key in zip(names, keys)}
    return result


@dataclass(frozen=True)
class InspectionResult:
    """Typed focused view of one claim, rule, object, concept, or graph node."""

    selector: str
    target_type: str
    target_id: str
    target: dict[str, object]
    related_claims: tuple[ExplanationClaim, ...]
    related_nodes: tuple[ExplanationNode, ...]
    related_edges: tuple[ExplanationEdge, ...]
    visual_spec: dict[str, object]

    def to_dict(self) -> dict[str, object]:
        return {
            "selector": self.selector,
            "target_type": self.target_type,
            "target_id": self.target_id,
            "target": self.target,
            "related_claims": [item.to_dict() for item in self.related_claims],
            "provenance": self.provenance(),
            "limitations": list(self.limitations()),
        }

    def summary(self) -> str:
        label = self.target.get("statement", self.target.get("label", self.target_id))
        return f"{self.target_type} {self.target_id}: {label}"

    def evidence(self) -> tuple[ExplanationNode, ...]:
        return self.related_nodes

    def limitations(self) -> tuple[str, ...]:
        values = [str(item) for item in self.target.get("limitations", ())]
        for claim in self.related_claims:
            values.extend(str(item) for item in claim.limitations)
        return tuple(dict.fromkeys(values))

    def provenance(self) -> dict[str, object]:
        return {
            "nodes": [item.to_dict() for item in self.related_nodes],
            "edges": [item.to_dict() for item in self.related_edges],
        }

    def audit(self) -> dict[str, object]:
        return {
            "selector": self.selector,
            "claim_ids": [item.claim_id for item in self.related_claims],
            "evidence_node_ids": [item.node_id for item in self.related_nodes],
            "relations": [item.relation for item in self.related_edges],
            "limitations": list(self.limitations()),
        }

    def visualize(
        self,
        *,
        view: str | None = None,
        backend: str = "matplotlib",
        output: str | Path | None = None,
    ):
        selected_view = view or ("rule_ablation" if self.selector.startswith("rule:") else "provenance")
        if backend == "matplotlib":
            from fuzzyxai.visualization.matplotlib_renderer import render_visual_spec
        elif backend == "plotly":
            from fuzzyxai.visualization.plotly_renderer import render_visual_spec
        else:
            raise ValueError(f"unsupported visualization backend: {backend}")
        return render_visual_spec(self.visual_spec, view=selected_view, output_path=output)


ExplanationInspection = InspectionResult


@dataclass(frozen=True)
class WhyNotExplanation:
    target_class: Any
    selected_prediction: Any
    status: str
    supports_selected: tuple[dict[str, Any], ...]
    supports_alternative: tuple[dict[str, Any], ...]
    key_difference: str
    required_change: str | None
    limitations: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class GlobalExplanationResult:
    task_type: str
    sample_count: int
    prediction_distribution: Mapping[str, int]
    global_evidence: Mapping[str, Any]
    capabilities: Mapping[str, Any]
    limitations: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class ModelComparisonResult:
    model_results: Mapping[str, ModelExplanationResult]
    prediction_agreement: bool
    reason_overlap: float | None
    disagreements: tuple[str, ...]
    limitations: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "models": {name: result.to_dict() for name, result in self.model_results.items()},
            "prediction_agreement": self.prediction_agreement,
            "reason_overlap": self.reason_overlap,
            "disagreements": list(self.disagreements),
            "limitations": list(self.limitations),
        }


@dataclass(frozen=True)
class ModelExplanationResult:
    """Public result returned by ``FuzzyXAI.wrap(...).explain(...)``."""

    prediction: ModelPrediction
    view_model: ExplanationViewModel

    @property
    def adapter_id(self) -> str:
        """Return the adapter that produced the canonical prediction."""

        return self.prediction.adapter_id

    @property
    def model_evidence(self) -> Mapping[str, Any]:
        """Expose serialized model evidence without leaking the adapter instance."""

        return self.view_model.model

    @property
    def action(self) -> str:
        return str(self.view_model.risk.get("action", "review"))

    @property
    def claims(self) -> tuple[ExplanationClaim, ...]:
        values = self.view_model.claims
        return tuple(ExplanationClaim.from_dict(item) for item in values) if isinstance(values, (list, tuple)) else tuple()

    @property
    def explanation_level(self) -> str:
        return str(self.view_model.explanation_level.get("level", "E0"))

    @property
    def available_channels(self) -> tuple[str, ...]:
        return tuple(self.view_model.explanation_level.get("available_channels", ()))

    @property
    def missing_channels(self) -> tuple[str, ...]:
        return tuple(self.view_model.explanation_level.get("missing_channels", ()))

    @property
    def native_channels(self) -> tuple[str, ...]:
        return tuple(self.view_model.explanation_level.get("native_channels", ()))

    @property
    def surrogate_channels(self) -> tuple[str, ...]:
        return tuple(self.view_model.explanation_level.get("surrogate_channels", ()))

    @property
    def object_representation(self) -> Mapping[str, Any] | None:
        """The raw explained object rendered back with its evidence overlaid.

        ``None`` when no representation could be built (no raw object and no
        data evidence at all); otherwise a mapping with ``modality`` ("text"
        or "tabular") discriminating which fields are populated. See
        ``visualization.spec.ObjectRepresentationSpec``.
        """

        return cast("Mapping[str, Any] | None", self.view_model.visual_spec.get("object_representation"))

    @property
    def similar_cases(self) -> tuple[Mapping[str, Any], ...]:
        """Reference-corpus objects most similar to the explained one, if any were measured.

        Empty when no reference corpus was registered (``FuzzyXAI.wrap(...,
        reference_data=...)`` or ``explain_one(..., reference_data=...)``) —
        never fabricated. Similarity is supporting/additional evidence, not
        a causal explanation of the prediction, unless the model adapter
        itself is prototype/exemplar-based (see ``evidence/human.py``'s
        ``_similar_statement``, which phrases it accordingly either way).
        """

        return tuple(cast("Sequence[Mapping[str, Any]]", self.view_model.visual_spec.get("similar_cases", ())))

    def to_dict(self, *, include_raw: bool = False, detail: str = "audit", audience: str = "domain_user") -> dict[str, Any]:
        """Serialize this result as one of three projections of the *same* canonical data.

        ``detail="audit"`` (the default, unchanged for backward compatibility)
        returns exactly what this method always returned: the full canonical
        payload (all claims, the explanation graph, every audience's
        ``HumanExplanation``, diagnostics, quality metrics, the full visual
        spec, trace/provenance).

        ``detail="standard"`` is the developer-facing projection: prediction,
        all claims, one audience's ``HumanExplanation``, similar cases,
        object representation, uncertainty (quality metrics), limitations,
        and the key provenance identifiers (not the full graph/trace).

        ``detail="compact"`` is the minimal projection for an application
        that just needs the answer: prediction, top supporting/contradicting
        evidence, similar cases, object representation, uncertainty,
        limitations, action, and minimal provenance refs.

        All three are read *only* from the already-computed
        ``self.view_model`` — none of them re-runs ``explain()``, so
        prediction/claims/similar-cases/action are guaranteed identical
        across tiers by construction, not by convention.

        ``include_raw=False`` (the default) strips the raw object text
        (``raw_object``/``raw_objects``) from every tier. This is *not* a
        general PII anonymizer — it only removes the one payload field that
        carries the complete original text; structured evidence derived from
        it (spans, offsets, tabular rows, feature values) is untouched.
        """

        if detail == "audit":
            return cast("dict[str, Any]", self.view_model.to_dict(include_raw=include_raw))
        if detail == "standard":
            return self._standard_dict(include_raw=include_raw, audience=audience)
        if detail == "compact":
            return self._compact_dict(include_raw=include_raw)
        raise ValueError(f"unsupported export detail level: {detail!r} (expected 'compact', 'standard', or 'audit')")

    def export_json(
        self,
        path: str | Path,
        *,
        include_raw: bool = False,
        detail: str = "audit",
        audience: str = "domain_user",
    ) -> Path:
        """Export this result as JSON. See ``to_dict`` for the ``detail``/``include_raw`` contract."""

        output = Path(path)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(
            json.dumps(self.to_dict(include_raw=include_raw, detail=detail, audience=audience), ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        return output

    def _redacted_object_representation(self, *, include_raw: bool) -> Mapping[str, Any] | None:
        representation = self.object_representation
        if representation is None or include_raw:
            return representation
        if representation.get("modality") == "text":
            return {**representation, "raw_excerpt": _RAW_TEXT_REDACTED, "highlighted_html": _RAW_TEXT_REDACTED}
        if representation.get("modality") == "image":
            return {**representation, "image_png_base64": _RAW_IMAGE_REDACTED}
        return representation

    def _provenance_summary(self) -> dict[str, Any]:
        trace = self.view_model.trace
        return {
            "adapter_id": trace.get("adapter_id"),
            "model_type": trace.get("model_type"),
            "model_fingerprint": trace.get("model_fingerprint"),
            "object_ids": trace.get("object_ids"),
            "dataset_version": trace.get("dataset_version"),
            "generated_at": trace.get("generated_at"),
            "input_sha256": trace.get("input_sha256"),
        }

    def _compact_dict(self, *, include_raw: bool) -> dict[str, Any]:
        decision_evidence = self.view_model.visual_spec.get("decision_evidence", {})
        supports = decision_evidence.get("supports", ()) if isinstance(decision_evidence, Mapping) else ()
        contradicts = decision_evidence.get("contradicts", ()) if isinstance(decision_evidence, Mapping) else ()
        limitations = decision_evidence.get("limitations", ()) if isinstance(decision_evidence, Mapping) else ()
        return {
            "detail": "compact",
            "prediction": {
                "value": self.prediction.predictions,
                "score": self.view_model.model.get("score"),
                "probabilities": self.prediction.probabilities,
            },
            "action": self.action,
            "supporting_evidence": list(supports)[:5],
            "contradicting_evidence": list(contradicts)[:5],
            "similar_cases": list(self.similar_cases),
            "object_representation": self._redacted_object_representation(include_raw=include_raw),
            "uncertainty": dict(self.view_model.quality_metrics),
            "limitations": list(limitations),
            "provenance": self._provenance_summary(),
        }

    def _standard_dict(self, *, include_raw: bool, audience: str) -> dict[str, Any]:
        decision_evidence = self.view_model.visual_spec.get("decision_evidence", {})
        limitations = decision_evidence.get("limitations", ()) if isinstance(decision_evidence, Mapping) else ()
        return {
            "detail": "standard",
            "prediction": {
                "value": self.prediction.predictions,
                "score": self.view_model.model.get("score"),
                "probabilities": self.prediction.probabilities,
            },
            "action": self.action,
            "claims": [claim.to_dict() for claim in self.claims],
            "human_explanation": {"audience": audience, **self.explain_for(audience).to_dict(include_technical_trace=False)},
            "similar_cases": list(self.similar_cases),
            "object_representation": self._redacted_object_representation(include_raw=include_raw),
            "uncertainty": dict(self.view_model.quality_metrics),
            "limitations": list(limitations),
            "provenance": self._provenance_summary(),
            "visual_metadata": {
                "schema_version": self.view_model.visual_spec.get("schema_version"),
                "overview": self.view_model.visual_spec.get("overview"),
            },
        }

    @property
    def explanation_graph(self) -> ExplanationGraph:
        return ExplanationGraph.from_dict(self.view_model.explanation_graph)

    def explain_for(
        self,
        audience: str = "domain_user",
        *,
        language: str = "ru",
    ) -> HumanExplanation:
        """Return typed, claim-grounded cards for the selected audience."""

        aliases = {"user": "domain_user", "expert": "ml_engineer", "audit": "auditor"}
        normalized = aliases.get(audience, audience)
        if language != "ru":
            raise ValueError("only the verified Russian human-explanation templates are available")
        payload = self.view_model.human_explanations.get(normalized)
        if not isinstance(payload, Mapping):
            raise ValueError(f"unknown or unavailable audience profile: {audience}")
        return HumanExplanation.from_dict(payload, graph=self.explanation_graph)

    def summary(
        self,
        audience: str = "domain_user",
        detail: str = "short",
        *,
        level: str | None = None,
    ) -> str:
        """Render audience-specific text; technical details require ``detail='full'``."""

        text = explanation_to_text(self.explain_for(level or audience), detail=detail)
        if detail == "short":
            # detail="full" already lists every similar case individually
            # (via HumanExplanation.details.similar_cases); this compact
            # digest of just the closest exemplar is what the short summary
            # is otherwise missing.
            digest = self._similar_cases_digest()
            if digest:
                text = text.rstrip() + "\n\n" + digest + "\n"
        return text

    def _similar_cases_digest(self) -> str:
        cases = self.similar_cases
        if not cases:
            return ""
        closest = cases[0]
        lines = ["## Похожие примеры", ""]
        reference_id = closest.get("reference_object_id", "?")
        lines.append(f"Наиболее близкий эталонный объект: {reference_id}.")
        score = closest.get("score")
        if isinstance(score, (int, float)):
            lines.append(f"Мера сходства: {float(score):.3f}.")
        rank, count = closest.get("reference_rank"), closest.get("reference_count")
        if isinstance(rank, int) and isinstance(count, int) and count > 0:
            lines.append(f"Место по близости: {rank} из {count} эталонных объектов.")
        matched = [str(item) for item in closest.get("matched_features", ())][:5]
        if matched:
            lines.append("")
            lines.append("Наиболее близкие характеристики:")
            lines.extend(f"- {name}" for name in matched)
        different = [str(item) for item in closest.get("different_features", ())][:5]
        if different:
            lines.append("")
            lines.append("Наиболее заметные различия:")
            lines.extend(f"- {name}" for name in different)
        lines.append("")
        lines.append(
            "Сходство является дополнительным сравнительным свидетельством "
            "и само по себе не устанавливает причину прогноза модели."
        )
        return "\n".join(lines)

    def overview(self) -> str:
        """Answer the five operational questions using claim-grounded text."""

        return self.summary(audience="domain_user", detail="short")

    def verbalize(
        self,
        *,
        backend: VerbalizationBackend | None = None,
        audience: str = "domain_user",
        detail: str = "short",
    ) -> str:
        """Rephrase the audience-appropriate summary through an optional SLM backend.

        With no backend (the default), returns the same text as ``summary()``
        — no new dependency or external service is ever required. Pass a
        backend such as ``fuzzyxai.verbalization.backends.OllamaBackend()`` to
        get a natural-language rephrasing; if the backend is unreachable or
        its output isn't grounded in the already-verified explanation, this
        silently falls back to ``summary()`` rather than raising or emitting
        unverified text. Use ``verbalize_detailed`` to see which path was
        taken.
        """

        return cast(str, self.verbalize_detailed(backend=backend, audience=audience, detail=detail).text)

    def verbalize_detailed(
        self,
        *,
        backend: VerbalizationBackend | None = None,
        audience: str = "domain_user",
        detail: str = "short",
    ) -> VerbalizationResult:
        from fuzzyxai.verbalization import SLMVerbalizer

        explanation = self.explain_for(audience)
        template_text = self.summary(audience=audience, detail=detail)
        return SLMVerbalizer(backend).run(explanation, template_text=template_text)

    def story(self) -> str:
        """Render the evidence route as data, training, knowledge, decision, action."""

        lines = [f"# История решения ({self.explanation_level})"]
        for stage in self.view_model.visual_spec.get("story", []):
            refs = ", ".join(stage.get("claim_refs", [])) or "нет claims"
            lines.append(f"\n## {stage.get('title')} [{stage.get('evidence_status')} / {stage.get('effect')}] ({refs})")
            facts = stage.get("facts", []) or ["Evidence для этапа отсутствует."]
            lines.extend(f"- {fact}" for fact in facts)
        return "\n".join(lines) + "\n"

    def inspect(self, selector: str) -> InspectionResult:
        """Inspect claim/rule/concept/object/evidence/diagnostic/action provenance."""

        prefix, separator, identifier = selector.partition(":")
        if selector == "action":
            prefix, identifier, separator = "action", "action", ":"
        if not separator or prefix not in {"claim", "rule", "concept", "object", "evidence", "node", "diagnostic", "action"}:
            raise ValueError("selector must identify claim, rule, concept, object, evidence, diagnostic, or action")
        graph = self.explanation_graph
        nodes = list(graph.nodes)
        edges = list(graph.edges)
        claims = list(self.claims)
        target: dict[str, object] | None = None
        anchors: set[str] = set()
        if prefix == "claim":
            normalized = identifier.upper().replace("C", "C-") if identifier.upper().startswith("C") and "-" not in identifier else identifier.upper()
            selected_claim = next((claim for claim in claims if claim.claim_id.upper() == normalized), None)
            target = selected_claim.to_dict() if selected_claim else None
            if selected_claim:
                anchors.add(f"claim:{selected_claim.claim_id}")
                anchors.update(str(item) for item in selected_claim.evidence_refs)
        elif prefix == "rule":
            target = next((rule for rule in self.view_model.layers.get("rules", []) if str(rule.get("rule_id")) == identifier), None)
            anchors.add(f"rule:{identifier}")
        elif prefix == "concept":
            node_id = f"concept:{identifier}"
            selected_node = next((node for node in nodes if node.node_id == node_id), None)
            target = selected_node.to_dict() if selected_node else None
            anchors.add(node_id)
        elif prefix == "object":
            candidates = {f"data:{identifier}", f"training:{identifier}"}
            selected_node = next((node for node in nodes if node.node_id in candidates), None)
            target = selected_node.to_dict() if selected_node else None
            anchors.update(candidates)
        elif prefix == "diagnostic":
            node_id = f"diagnostic:{identifier}"
            selected_node = next((node for node in nodes if node.node_id == node_id), None)
            target = selected_node.to_dict() if selected_node else None
            anchors.add(node_id)
        elif prefix == "action":
            selected_node = next((node for node in nodes if node.node_id == "action"), None)
            target = selected_node.to_dict() if selected_node else None
            anchors.add("action")
        else:
            node_id = identifier if prefix == "node" else selector.removeprefix("evidence:")
            selected_node = next((node for node in nodes if node.node_id == node_id), None)
            target = selected_node.to_dict() if selected_node else None
            anchors.add(node_id)
        if target is None:
            raise KeyError(f"unknown inspection target: {selector}")
        related_edges = [edge for edge in edges if edge.source in anchors or edge.target in anchors]
        related_ids = set(anchors)
        for edge in related_edges:
            related_ids.update((edge.source, edge.target))
        related_nodes = [node for node in nodes if node.node_id in related_ids]
        related_claims = [
            claim
            for claim in claims
            if f"claim:{claim.claim_id}" in related_ids
            or any(str(ref) in related_ids for ref in claim.evidence_refs)
        ]
        return InspectionResult(selector, prefix, identifier, target, tuple(related_claims), tuple(related_nodes), tuple(related_edges), dict(self.view_model.visual_spec))

    def audit(self) -> dict[str, Any]:
        """Return full provenance and channel disclosure without presentation text."""

        return {
            "explanation_level": dict(self.view_model.explanation_level),
            "claims": [claim.to_dict() for claim in self.claims],
            "graph": dict(self.view_model.explanation_graph),
            "diagnostics": list(self.view_model.diagnostics),
            "action": self.action,
            "trace": dict(self.view_model.trace),
            "quality_metrics": dict(self.view_model.quality_metrics),
        }

    def diagnose(
        self,
        *,
        repair_mode: str = "plan",
        repair_context: RepairExecutionContext | None = None,
        audience: str = "user",
    ) -> DiagnosticReport:
        """Diagnose the serialized explanation graph without inferring model error."""

        from fuzzyxai.diagnostics.service import DiagnosticService

        graph = self.explanation_graph
        route = {
            "route_id": str(self.view_model.trace.get("trace_id", "explanation")),
            "schema_version": graph.schema_version,
            "nodes": [
                {
                    "node_id": node.node_id,
                    "node_type": node.node_type,
                    "observed_attributes": dict(node.payload),
                    "mandatory": True,
                    "repairable": node.node_type not in {"prediction", "action"},
                    "evidence_refs": tuple(node.evidence_refs),
                }
                for node in graph.nodes
            ],
            "edges": [
                {
                    "edge_id": f"edge:{index}:{edge.source}->{edge.target}",
                    "source": edge.source,
                    "target": edge.target,
                    "relation": edge.relation if edge.relation in {
                        "produces",
                        "consumes",
                        "derived_from",
                        "explains",
                        "calibrates",
                        "transforms",
                        "validates",
                        "aggregates",
                        "reduces",
                        "certifies",
                        "blocks",
                    } else "derived_from",
                    "mandatory": True,
                    "registered_contract": {"linked": True},
                    "observed_contract": {"linked": True},
                    "evidence_refs": tuple(edge.evidence_refs),
                }
                for index, edge in enumerate(graph.edges)
            ],
            "metadata": {"missing_evidence": tuple(graph.missing_evidence)},
            "contracts": [
                {
                    "contract_id": f"evidence:{index}:{channel}",
                    "kind": "equals",
                    "subject_id": "trace:missing_evidence",
                    "field": channel,
                    "expected": True,
                    "repairable": True,
                    "category": "provenance",
                    "source_nodes": ("trace:missing_evidence",),
                }
                for index, channel in enumerate(graph.missing_evidence)
            ],
        }
        return DiagnosticService().diagnose(
            route=route,
            repair_mode=repair_mode,
            repair_context=repair_context,
            audience=audience,
        )

    def quality_report(self) -> ExplanationQualityReport:
        task_type = str(self.prediction.metadata.get("task_type", ""))
        return build_quality_report(self.view_model.quality_metrics, regression=task_type == TaskType.REGRESSION.value)

    def capability_report(self) -> dict[str, Any]:
        return {
            "adapter_id": self.view_model.trace.get("adapter_id"),
            "model_type": self.view_model.trace.get("model_type"),
            "task_type": self.prediction.metadata.get("task_type"),
            "capabilities": dict(self.view_model.trace.get("adapter_capabilities", {})),
            "resolution": dict(self.view_model.trace.get("adapter_resolution", {})),
            "planner": dict(self.view_model.trace.get("explanation_plan", {})),
        }

    def why_not(self, target_class: Any) -> WhyNotExplanation:
        predictions = self.prediction.predictions
        selected = predictions[0] if isinstance(predictions, list) and predictions else predictions
        contributions = self.view_model.model.get("contributions", {})
        if not isinstance(contributions, Mapping) or not contributions:
            return WhyNotExplanation(
                target_class,
                selected,
                "insufficient_evidence",
                (),
                (),
                "Локальные вклады для сравнения классов недоступны.",
                None,
                ("Alternative-class evidence was not measured.",),
            )
        ranked = sorted(((str(name), float(value)) for name, value in contributions.items()), key=lambda item: abs(item[1]), reverse=True)
        supports_selected = tuple({"feature": name, "value": value} for name, value in ranked if value >= 0)[:3]
        supports_alternative = tuple({"feature": name, "value": value} for name, value in ranked if value < 0)[:3]
        difference = (
            f"Выбранный класс {selected} сильнее поддержан положительными локальными вкладами; "
            f"для класса {target_class} доступны только противоречащие факторы."
        )
        counterfactuals = self.view_model.layers.get("counterfactuals", [])
        checked = next((item for item in counterfactuals if item.get("target_prediction") == target_class), None)
        required_change = None if checked is None else str(checked.get("changed_features") or checked.get("changed_rules"))
        limitations = () if checked is not None else ("A class-changing intervention was not measured; no required change is asserted.",)
        return WhyNotExplanation(target_class, selected, "supported", supports_selected, supports_alternative, difference, required_change, limitations)

    def plot(
        self,
        output_path: str | Path | None = None,
        *,
        kind: str = "dashboard",
        backend: str = "matplotlib",
    ):
        return self.visualize(view=kind, backend=backend, output=output_path)

    def visualize(
        self,
        *,
        view: str = "explanation_story",
        kind: str | None = None,
        backend: str = "matplotlib",
        output: str | Path | None = None,
        output_path: str | Path | None = None,
    ):
        selected_view = kind or view
        selected_output = output if output is not None else output_path
        if backend == "matplotlib":
            from fuzzyxai.visualization.matplotlib_renderer import render_visual_spec
        elif backend == "plotly":
            from fuzzyxai.visualization.plotly_renderer import render_visual_spec
        else:
            raise ValueError(f"unsupported visualization backend: {backend}")
        return render_visual_spec(self.view_model.visual_spec, view=selected_view, output_path=selected_output)

    def export_html(self, path: str | Path) -> Path:
        """Export a self-contained evidence report without recomputing metrics."""

        output = Path(path)
        output.parent.mkdir(parents=True, exist_ok=True)
        summaries = "\n".join(
            f"<section><h2>{escape(audience.title())}</h2><pre>{escape(self.summary(audience))}</pre></section>"
            for audience in ("domain_user", "ml_engineer", "researcher", "auditor")
            if audience in self.view_model.human_explanations
        )
        output.write_text(
            "<!doctype html><html lang='ru'><meta charset='utf-8'>"
            "<title>FuzzyXAI explanation</title>"
            "<style>body{font:16px Georgia,serif;max-width:1100px;margin:40px auto;color:#1e2a32}"
            "section{border-top:1px solid #ccd4d8;padding:18px 0}pre{white-space:pre-wrap;font:15px Georgia,serif}"
            "details{margin-top:24px}</style><body><h1>FuzzyXAI explanation report</h1>"
            + summaries
            + "<details><summary>Canonical JSON</summary><pre>"
            + escape(self.view_model.to_json())
            + "</pre></details></body></html>\n",
            encoding="utf-8",
        )
        return output


class FuzzyXAI:
    """Runtime facade for using FuzzyXAI as an installable framework."""

    def __init__(
        self,
        plan: ExplainPlan | None = None,
        model_adapter: ModelAdapter | None = None,
        resolution_report: AdapterResolutionReport | None = None,
        *,
        reference_data: Any | None = None,
        reference_labels: Any | None = None,
        reference_ids: list[str] | None = None,
    ):
        self.plan = plan or ExplainPlan.default()
        self._model_adapter = model_adapter
        self._resolution_report = resolution_report
        # A reference corpus registered once at wrap() time, used by default
        # for similar-case evidence on every explain_one() call so the
        # caller doesn't have to repeat reference_data/reference_labels on
        # every call. Still overridable per-call.
        self._reference_data = reference_data
        self._reference_labels = reference_labels
        self._reference_ids = reference_ids

    @property
    def model_adapter(self) -> ModelAdapter:
        if self._model_adapter is None:
            raise RuntimeError("no model adapter is configured")
        return self._model_adapter

    @classmethod
    def wrap(
        cls,
        model: Any,
        *,
        adapter: str | ModelAdapter = "auto",
        explain_plan: ExplainPlan | None = None,
        task: str | TaskType = "auto",
        output_decoder: Any = None,
        reference_data: Any | None = None,
        reference_labels: Any | None = None,
        reference_ids: list[str] | None = None,
    ) -> FuzzyXAI:
        """Wrap a supported model using capability-based adapter resolution.

        ``reference_data``/``reference_labels``/``reference_ids`` register a
        reference corpus once, here, instead of on every ``explain_one()``
        call — when present, similar-case evidence is produced by default
        (see ``explain()``'s ``include_similar_cases``).
        """

        resolved, report = resolve_model_adapter_v2(
            model,
            task=task,
            adapter=adapter,
            output_decoder=output_decoder,
        )
        return cls(
            plan=explain_plan,
            model_adapter=resolved,
            resolution_report=report,
            reference_data=reference_data,
            reference_labels=reference_labels,
            reference_ids=reference_ids,
        )

    @classmethod
    def register_adapter(
        cls,
        *,
        adapter_class: Any,
        predicate: Any,
        priority: int = 100,
        name: str | None = None,
    ) -> None:
        del cls
        MODEL_ADAPTER_REGISTRY.register(adapter_class=adapter_class, predicate=predicate, priority=priority, name=name)

    def capability_report(self) -> dict[str, Any]:
        capabilities = self.model_adapter.capabilities()
        input_schema = self.model_adapter.input_schema() if isinstance(self.model_adapter, ModelAdapterV2) else None
        output_schema = self.model_adapter.output_schema() if isinstance(self.model_adapter, ModelAdapterV2) else None
        return {
            "adapter_id": self.model_adapter.adapter_id,
            "model_family": str(getattr(self.model_adapter, "model_family", "legacy")),
            "task_type": str(getattr(getattr(self.model_adapter, "task_type", None), "value", "unknown")),
            "capabilities": capabilities.to_dict(),
            "input_schema": asdict(input_schema) if input_schema else None,
            "output_schema": {**asdict(output_schema), "task_type": output_schema.task_type.value} if output_schema else None,
            "resolution": self._resolution_report.to_dict() if self._resolution_report else None,
        }

    def diagnose(
        self,
        *,
        route: object,
        repair_mode: str = "plan",
        repair_context: RepairExecutionContext | None = None,
        audience: str = "user",
    ) -> DiagnosticReport:
        """Diagnose route integrity independently from prediction correctness."""

        from fuzzyxai.diagnostics.service import DiagnosticService

        return DiagnosticService().diagnose(
            route=route,
            repair_mode=repair_mode,
            repair_context=repair_context,
            audience=audience,
        )

    def diagnose_batch(
        self,
        *,
        routes: object,
        repair_mode: str = "plan",
    ) -> BatchDiagnosticReport:
        """Diagnose multiple routes; external repair execution is intentionally disabled."""

        from fuzzyxai.diagnostics.service import DiagnosticService

        return DiagnosticService().diagnose_batch(routes=routes, repair_mode=repair_mode)

    def explain(
        self,
        inputs: Any,
        *,
        evidence: Mapping[str, Any] | None = None,
        object_ids: list[str] | None = None,
        feature_names: list[str] | None = None,
        reference_data: Any | None = None,
        reference_ids: list[str] | None = None,
        reference_labels: list[Any] | None = None,
        training_run: TrainingRunAnalysis | None = None,
        include_similar_cases: bool | None = None,
        include_counterfactuals: bool = False,
        include_training_trace: bool = False,
        include_model_knowledge: bool = True,
        additional_evidence: ExplanationEvidence | None = None,
        dataset_version: str = "unversioned",
        run_parameters: Mapping[str, Any] | None = None,
        raw_objects: Sequence[Any] | None = None,
        region_masks: Mapping[str, Sequence[Sequence[bool]]] | None = None,
    ) -> ModelExplanationResult:
        """Explain a prediction using only supplied, auditable operator evidence.

        A prediction can always be returned. Gamma, Delta, and rho are computed
        only when their typed evidence sections are present; otherwise the
        result is marked for review instead of receiving synthetic values.

        ``raw_objects`` optionally carries the original, unvectorized objects
        alongside their numeric feature rows, so the explanation package can
        show evidence overlaid on the object itself rather than only on
        abstract feature names. When supplied, its length must equal the
        number of rows/object_ids (a mismatch raises ``ValueError``). Only
        the first object is used for highlighting, matching the existing
        single-object scope of ``include_similar_cases``/
        ``include_counterfactuals``. A ``str`` triggers text highlighting;
        a 2D/3D array-like object (numpy array or anything PIL-like) is
        treated as an image and triggers image representation instead; any
        other type is disclosed as unsupported rather than guessed at, and
        the presentation layer falls back to a tabular feature/value/
        contribution view built from already-collected data evidence.
        Absent by default: no raw object means no highlighting/image
        evidence, never a fabricated one.

        ``region_masks`` optionally names boolean pixel masks (each matching
        the image's own height/width) for the first object, when it is an
        image — e.g. from a segmentation model or manual annotation.
        FuzzyXAI has no built-in per-pixel attribution method, so without
        this the image representation still shows the image itself but with
        no regions, honestly disclosed as a limitation rather than a
        fabricated heatmap.
        """

        if self._model_adapter is None:
            raise RuntimeError("FuzzyXAI.explain requires FuzzyXAI.wrap(model, ...)")
        # A reference corpus registered on FuzzyXAI.wrap(...) applies by
        # default; an explicit per-call value still wins.
        if reference_data is None:
            reference_data = self._reference_data
        if reference_labels is None:
            reference_labels = self._reference_labels
        if reference_ids is None:
            reference_ids = self._reference_ids
        evidence = dict(evidence or {})
        prediction = self._model_adapter.predict(inputs)
        score = prediction.primary_score()
        diagnostics: list[dict[str, Any]] = []
        rows = _as_rows(inputs)
        reference_rows = _as_rows(reference_data) if reference_data is not None else None
        if include_similar_cases is None:
            # Similar-case evidence is produced by default whenever a
            # reference corpus is actually available (from wrap() or this
            # call) — the user should not have to separately remember to
            # pass include_similar_cases=True on top of reference_data.
            # Absent a reference corpus, this correctly resolves to False:
            # no fabricated similarity evidence.
            include_similar_cases = reference_rows is not None
        names = list(feature_names or self._model_adapter.feature_names() or _default_feature_names(rows))
        ids = list(object_ids or [f"object_{index}" for index in range(len(rows))])
        internal_evidence = _with_feature_names(self._model_adapter.extract_internal_evidence(inputs), names)
        adapter_capabilities = self._model_adapter.capabilities()
        planner = ExplanationPlanner()
        planner_decision = planner.plan(
            adapter_capabilities if isinstance(adapter_capabilities, ModelCapabilities) else ModelCapabilities(
                predict=True,
                predict_proba=adapter_capabilities.get("predict_proba"),
                local_contributions=adapter_capabilities.get("feature_importance"),
                native_rules=adapter_capabilities.get("rules"),
            ),
            budget=str((run_parameters or {}).get("budget", "standard")),
            regression=prediction.metadata.get("task_type") == TaskType.REGRESSION.value,
        )
        internal_descriptors = internal_evidence.get("evidence_descriptors", [])
        surrogate_fidelity = internal_evidence.get("surrogate_fidelity")
        low_fidelity_surrogate = any(
            isinstance(item, Mapping) and item.get("origin") == "surrogate" and item.get("name") == "local_contributions"
            for item in internal_descriptors
        ) and (surrogate_fidelity is None or float(surrogate_fidelity) < (0.8 if prediction.metadata.get("task_type") == TaskType.REGRESSION.value else 0.9))
        if low_fidelity_surrogate:
            internal_evidence.pop("contributions", None)
            diagnostics.append(
                {
                    "code": "D_surrogate_fidelity",
                    "reason": "surrogate local explanation fidelity is missing or below the configured threshold",
                    "severity": "warning",
                }
            )

        alignment_data = evidence.get("alignment")
        alignment = None
        if isinstance(alignment_data, Mapping):
            alignment = compute_operator_alignment(
                AlignmentInput(
                    components=dict(alignment_data["components"]),
                    weights=dict(alignment_data["weights"]),
                    gamma_max=float(alignment_data.get("gamma_max", self.plan.gamma_critical)),
                    delta_t=float(alignment_data.get("delta_t", 0.0)),
                    delta_max=float(alignment_data.get("delta_max", self.plan.delta_critical)),
                )
            )
            if not alignment.certified:
                diagnostics.append(
                    {
                        "code": "D_ij_alignment",
                        "reason": "alignment exceeds the configured gamma or transition-loss boundary",
                        "severity": "error",
                    }
                )
        else:
            diagnostics.append({"code": "D_k_alignment_missing", "reason": "alignment evidence was not supplied", "severity": "warning"})

        reduction_data = evidence.get("reduction")
        reduction = None
        if isinstance(reduction_data, Mapping):
            reduction = compute_operator_reduction(
                ReductionInput(
                    components=dict(reduction_data["components"]),
                    weights=dict(reduction_data["weights"]),
                    delta_max=float(reduction_data.get("delta_max", self.plan.delta_critical)),
                    kappa_delta=float(reduction_data.get("kappa_delta", 1.0)),
                )
            )
            if not reduction.allowed:
                diagnostics.append(
                    {
                        "code": "D_reduction",
                        "reason": "representation reduction exceeds delta_max",
                        "severity": "error",
                    }
                )
        else:
            diagnostics.append({"code": "D_k_reduction_missing", "reason": "reduction evidence was not supplied", "severity": "warning"})

        risk_data = evidence.get("risk")
        risk = None
        if isinstance(risk_data, Mapping):
            thresholds = dict(
                risk_data.get(
                    "thresholds",
                    {
                        "theta_1": self.plan.rho_accept,
                        "theta_2": self.plan.rho_warning,
                        "theta_3": self.plan.rho_audit,
                        "theta_4": self.plan.rho_critical,
                    },
                )
            )
            risk = observe_operator_risk(
                RiskInput(
                    components=dict(risk_data["components"]),
                    weights=dict(risk_data["weights"]),
                    thresholds=thresholds,
                    chi_r_crit=int(risk_data.get("chi_r_crit", 0)),
                )
            )
            if risk.chi_r_crit:
                diagnostics.append(
                    {
                        "code": "D_risk_critical",
                        "reason": "critical risk rupture forbids automatic acceptance",
                        "severity": "critical",
                    }
                )
        else:
            diagnostics.append({"code": "D_k_risk_missing", "reason": "risk evidence was not supplied", "severity": "warning"})

        action = risk.action if risk is not None else "review"
        structural_failure = (alignment is not None and not alignment.certified) or (reduction is not None and not reduction.allowed)
        if structural_failure and action != "block":
            action = "review"
        route = [
            {"id": "model", "label": "Model", "status": "passed"},
            {"id": "adapter", "label": prediction.adapter_id, "status": "passed"},
            {"id": "alignment", "label": "T_ij", "status": "passed" if alignment and alignment.certified else "warning"},
            {"id": "reduction", "label": "Delta", "status": "passed" if reduction and reduction.allowed else "warning"},
            {"id": "risk", "label": "Risk", "status": "blocked" if action == "block" else ("passed" if risk else "warning")},
            {"id": "action", "label": action, "status": "blocked" if action == "block" else "passed"},
        ]
        plan_json = json.dumps(self.plan.to_dict(), ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        data_evidence = collect_data_evidence(
            rows,
            object_ids=ids,
            feature_names=names,
            reference_values=reference_rows,
            source_trace={"adapter_id": prediction.adapter_id, "model_fingerprint": self._model_adapter.model_fingerprint()},
        )
        missing: list[str] = []
        required_missing: list[str] = []
        training_evidence = []
        subgroup_evidence = []
        if include_training_trace:
            if training_run is None:
                missing.append("training_trace")
                required_missing.append("training_trace")
            else:
                training_evidence = [training_run.traces[item] for item in ids if item in training_run.traces]
                subgroup_evidence = list(training_run.subgroups)
                if not training_evidence:
                    missing.append("training_trace_for_requested_objects")
                    required_missing.append("training_trace_for_requested_objects")
        rules = []
        concepts = []
        if include_model_knowledge:
            rules = extract_rules(
                self._model_adapter,
                feature_names=names,
                model_version=self._model_adapter.model_fingerprint()[:12],
            )
            if reference_rows is not None and reference_labels is not None:
                concepts = build_class_concepts(
                    reference_rows,
                    reference_labels,
                    feature_names=names,
                    object_ids=reference_ids,
                    rules=rules,
                )
        contributions = dict(evidence.get("contributions", internal_evidence.get("contributions", {})))
        contribution_method = evidence.get("contribution_method", internal_evidence.get("contribution_method"))
        raw_activated_rules = evidence.get("activated_rules", internal_evidence.get("activated_rules"))
        fuzzy_rule_activations = (
            collect_fuzzy_rule_activations(raw_activated_rules, object_id=ids[0])
            if isinstance(raw_activated_rules, Sequence) and raw_activated_rules
            else []
        )
        if include_model_knowledge and not rules and not fuzzy_rule_activations:
            # Rule-based knowledge is genuinely absent only when neither
            # channel produced anything — a fuzzy/rule model that supplies
            # activated_rules must not also be told "model rules are
            # missing" in the same result.
            missing.append("model_rules_or_concepts")
        text_highlights = []
        image_representations = []
        if raw_objects is not None:
            if len(raw_objects) != len(rows):
                raise ValueError(f"raw_objects has {len(raw_objects)} entries but {len(rows)} objects were supplied to explain(); they must match one-to-one")
            raw_object = raw_objects[0]
            if isinstance(raw_object, str):
                if not contributions:
                    missing.append("text_highlight_contributions")
                else:
                    text_highlights = [find_text_highlight_spans(raw_object, contributions, object_id=ids[0])]
            elif is_image_like(raw_object):
                image_representations = [find_image_regions(raw_object, contributions, object_id=ids[0], region_masks=region_masks)]
            else:
                # Neither a string nor image-shaped — disclosed as unused
                # rather than guessed at; build_visual_spec's tabular
                # fallback still gives an honest representation from the
                # already-collected data evidence.
                missing.append("text_highlight_unsupported_raw_object_type")
        similar_cases = []
        if include_similar_cases:
            if reference_rows is None:
                missing.append("similar_case_reference_data")
                required_missing.append("similar_case_reference_data")
            else:
                similar_cases = find_similar_tabular_cases(
                    rows[0],
                    reference_rows,
                    query_object_id=ids[0],
                    reference_ids=reference_ids,
                    feature_names=names,
                    reference_labels=reference_labels,
                )
        counterfactuals = []
        if include_counterfactuals:
            if reference_rows is None:
                missing.append("counterfactual_reference_data")
                required_missing.append("counterfactual_reference_data")
            else:
                counterfactuals = find_tabular_counterfactuals(
                    self._model_adapter,
                    rows[0],
                    reference_rows,
                    feature_names=names,
                )
                if not counterfactuals:
                    missing.append("class_changing_counterfactual")
                    required_missing.append("class_changing_counterfactual")
        if required_missing and action != "block":
            action = "insufficient_evidence"
            route[-1] = {"id": "action", "label": action, "status": "warning"}
        additional = additional_evidence or ExplanationEvidence()
        explanation_evidence = ExplanationEvidence(
            data=[*data_evidence, *additional.data],
            training=[*training_evidence, *additional.training],
            subgroups=[*subgroup_evidence, *additional.subgroups],
            rules=[*rules, *additional.rules],
            concepts=[*concepts, *additional.concepts],
            similar_cases=[*similar_cases, *additional.similar_cases],
            counterfactuals=[*counterfactuals, *additional.counterfactuals],
            text_highlights=[*text_highlights, *additional.text_highlights],
            image_representations=[*image_representations, *additional.image_representations],
            fuzzy_rule_activations=[*fuzzy_rule_activations, *additional.fuzzy_rule_activations],
            missing=list(dict.fromkeys([*missing, *additional.missing])),
        )
        prediction_payload = {
            **prediction.to_dict(),
            "score": score,
            "contributions": contributions,
            "contribution_method": contribution_method,
        }
        claims = build_explanation_claims(
            explanation_evidence,
            prediction=prediction_payload,
            diagnostics=diagnostics,
            action=action,
        )
        graph = build_explanation_graph(
            explanation_evidence,
            prediction=prediction_payload,
            diagnostics=diagnostics,
            action=action,
            claims=claims,
        )
        explanation_level = determine_explanation_level(
            explanation_evidence,
            contribution_method=str(contribution_method) if contribution_method else None,
            operator_channels={
                "alignment": alignment is not None,
                "reduction": reduction is not None,
                "risk": risk is not None,
            },
        )
        human = {
            audience: compose_human_explanation(
                claims,
                graph,
                action=action,
                audience=audience,
                evidence=explanation_evidence,
                domain_language=self.plan.domain_language,
                task_type=str(prediction.metadata.get("task_type", "")) or None,
            ).to_dict(include_technical_trace=False)
            for audience in ("domain_user", "ml_engineer", "researcher", "auditor")
        }
        human.update(
            {
                "user": human["domain_user"],
                "expert": human["ml_engineer"],
                "audit": human["auditor"],
            }
        )
        quality_metrics = evaluate_explanation_quality(
            explanation_evidence,
            graph,
            contributions=contributions,
            supplied_metrics={
                **dict(evidence.get("quality_metrics", {})),
                **({"fidelity": float(surrogate_fidelity)} if surrogate_fidelity is not None else {}),
                **(
                    {"reconstruction_error": float(internal_evidence["reconstruction_error"])}
                    if internal_evidence.get("reconstruction_error") is not None
                    else {}
                ),
            },
        )
        visual_spec = build_visual_spec(
            explanation_evidence,
            claims,
            graph,
            prediction=prediction_payload,
            action=action,
            contributions=contributions,
            explanation_level=explanation_level.to_dict(),
        )
        view_model = ExplanationViewModel(
            model={
                **prediction.to_dict(),
                "score": score,
                "contributions": contributions,
                "contribution_method": contribution_method,
                "contribution_limitations": list(internal_evidence.get("limitations", [])),
            },
            fuzzy={"memberships": dict(evidence.get("memberships", {}))},
            route=route,
            disagreement={
                "components": dict(alignment_data.get("components", {})) if isinstance(alignment_data, Mapping) else {},
                "gamma": alignment.gamma if alignment else None,
                "delta_t": alignment.delta_t if alignment else None,
                "delta": reduction.delta if reduction else None,
                "r_delta": reduction.r_delta if reduction else None,
            },
            risk={
                "components": dict(risk_data.get("components", {})) if isinstance(risk_data, Mapping) else {},
                "rho": risk.rho if risk else None,
                "chi_r_crit": risk.chi_r_crit if risk else None,
                "action": action,
            },
            diagnostics=diagnostics,
            claims=[claim.to_dict() for claim in claims],
            narrative=" ".join(claim.statement for claim in claims if claim.claim_type in {"prediction", "recommended_action"}),
            trace={
                "adapter_id": prediction.adapter_id,
                "model_type": prediction.model_type,
                "model_fingerprint": self._model_adapter.model_fingerprint(),
                "adapter_capabilities": adapter_capabilities.to_dict(),
                "adapter_resolution": self._resolution_report.to_dict() if self._resolution_report else {},
                "explanation_plan": planner_decision.to_dict(),
                "object_ids": ids,
                "dataset_version": dataset_version,
                "input_sha256": _payload_sha256(rows),
                "generated_at": datetime.now(UTC).isoformat(),
                "run_parameters": dict(run_parameters or {}),
                "missing_evidence": missing,
                "explain_plan_sha256": sha256(plan_json.encode("utf-8")).hexdigest(),
            },
            layers=explanation_evidence.to_dict(),
            explanation_graph=graph.to_dict(),
            human_explanations=human,
            quality_metrics=quality_metrics,
            explanation_level=explanation_level.to_dict(),
            visual_spec=visual_spec.to_dict(),
        )
        return ModelExplanationResult(prediction=prediction, view_model=view_model)

    def explain_one(
        self,
        input_object: Any,
        *,
        object_id: str = "object_0",
        include_similar_cases: bool | None = None,
        include_counterfactuals: bool = False,
        include_training_trace: bool = False,
        raw_object: Any | None = None,
        **kwargs: Any,
    ) -> ModelExplanationResult:
        """Explain one object while preserving its identifier in every layer.

        ``raw_object`` optionally carries the original, unvectorized object
        (a raw text string, or a 2D/3D image array) so the explanation
        package can show evidence overlaid on the object itself. See
        ``explain()`` for the full contract (including ``region_masks`` for
        images); here it is a single value, not a sequence, since this
        method always explains exactly one object.
        """

        return self.explain(
            _as_rows(input_object),
            object_ids=[object_id],
            include_similar_cases=include_similar_cases,
            include_counterfactuals=include_counterfactuals,
            include_training_trace=include_training_trace,
            raw_objects=None if raw_object is None else [raw_object],
            **kwargs,
        )

    def explain_batch(
        self,
        inputs: Any,
        *,
        object_ids: list[str] | None = None,
        raw_objects: Sequence[Any] | None = None,
        **kwargs: Any,
    ) -> ModelExplanationResult:
        rows = _as_rows(inputs)
        ids = object_ids or [f"object_{index}" for index in range(len(rows))]
        return self.explain(rows, object_ids=ids, raw_objects=raw_objects, **kwargs)

    def explain_global(
        self,
        reference_data: Any,
        reference_labels: Any = None,
        *,
        feature_names: list[str] | None = None,
    ) -> GlobalExplanationResult:
        rows = _as_rows(reference_data)
        prediction = self.model_adapter.predict(rows)
        values = prediction.predictions if isinstance(prediction.predictions, list) else [prediction.predictions]
        distribution: dict[str, int] = {}
        for value in values:
            key = str(value)
            distribution[key] = distribution.get(key, 0) + 1
        if isinstance(self.model_adapter, ModelAdapterV2):
            context = ExplanationContext(
                reference_data=rows,
                reference_labels=reference_labels,
                feature_names=tuple(feature_names or self.model_adapter.feature_names()),
            )
            global_evidence = self.model_adapter.extract_global_evidence(context)
            payload = dict(global_evidence.channels)
            limitations = global_evidence.limitations
            task_type = self.model_adapter.task_type.value
        else:
            payload = {}
            limitations = ("Legacy adapter exposes no typed global evidence.",)
            task_type = "unknown"
        return GlobalExplanationResult(task_type, len(rows), distribution, payload, self.model_adapter.capabilities().to_dict(), limitations)

    @classmethod
    def compare_models(
        cls,
        models: Mapping[str, Any],
        *,
        item: Any,
        reference_data: Any = None,
        reference_labels: Any = None,
        task: str | TaskType = "auto",
        feature_names: list[str] | None = None,
    ) -> ModelComparisonResult:
        results = {
            name: cls.wrap(model, task=task).explain_one(
                item,
                object_id="comparison_object",
                reference_data=reference_data,
                reference_labels=reference_labels,
                feature_names=feature_names,
            )
            for name, model in models.items()
        }
        predictions = []
        reason_sets = []
        for result in results.values():
            value = result.prediction.predictions
            predictions.append(value[0] if isinstance(value, list) and value else value)
            reason_sets.append({item.subject_id for item in result.claims if item.claim_type == "feature_contribution"})
        agreement = len({str(item) for item in predictions}) <= 1
        if len(reason_sets) < 2 or not any(reason_sets):
            overlap = None
        else:
            union = set.union(*reason_sets)
            intersection = set.intersection(*reason_sets)
            overlap = len(intersection) / len(union) if union else None
        disagreements = tuple(
            f"{name}: prediction={prediction}"
            for (name, _), prediction in zip(results.items(), predictions)
            if not agreement
        )
        return ModelComparisonResult(
            results,
            agreement,
            overlap,
            disagreements,
            ("Reason agreement compares disclosed local evidence channels; missing channels are not imputed.",),
        )

    def observe_training(
        self,
        *,
        train_data: Any = None,
        val_data: Any = None,
        history: Mapping[str, Any],
        checkpoints: Any = None,
    ) -> TrainingRunAnalysis:
        """Create auditable object trajectories and subgroup diagnostics."""

        del train_data, val_data, checkpoints
        object_history = history.get("objects", {})
        traces = {str(object_id): build_object_trace(str(object_id), metrics) for object_id, metrics in object_history.items()}
        subgroups = []
        if history.get("global_metric") and history.get("subgroup_metrics"):
            subgroups = detect_subgroup_averaging(
                global_metric=history["global_metric"],
                subgroup_metrics=history["subgroup_metrics"],
                subgroup_objects=history.get("subgroup_objects"),
                subgroup_rule_history=history.get("subgroup_rule_history"),
                embedding_spread=history.get("embedding_spread"),
            )
        rules = extract_rules(
            self._model_adapter,
            feature_names=self._model_adapter.feature_names(),
            model_version=self._model_adapter.model_fingerprint()[:12],
        ) if self._model_adapter is not None else []
        return TrainingRunAnalysis(traces=traces, subgroups=subgroups, rules=rules)

    def run(self, adapted_input: AdaptedInput) -> OperatorRoute:
        return build_route(adapted_input)

    def run_payload(self, payload: dict, adapter: BaseAdapter) -> OperatorRoute:
        validation = adapter.validate_payload(payload)
        if not validation.valid:
            raise ValueError("; ".join(validation.errors))
        return self.run(adapter.to_adapted_input(payload))

    def verify(self, route: OperatorRoute) -> VerificationResult:
        return verify_proof_trace(build_proof_trace(route))

    def export_package(self, route: OperatorRoute, output_dir: str | Path) -> dict[str, Path]:
        output = Path(output_dir)
        output.mkdir(parents=True, exist_ok=True)
        trace = build_proof_trace(route)
        verification = verify_proof_trace(trace)
        paths = {
            "route": save_route_json(route, output / "route.json"),
            "proof_trace": save_proof_trace_json(trace, output / "proof_trace.json"),
            "dashboard": render_dashboard(route, output / "operator_dashboard.png"),
        }
        paths.update(write_traceability_artifacts(route, trace, verification, output))
        (output / "summary.json").write_text(
            json.dumps(
                {
                    "scenario_id": route.scenario_id,
                    "action_id": route.final_action_id or route.final_action,
                    "diagnostic_id": route.final_diagnostic_id or route.computed_result.get("diagnostic_id"),
                    "verifier": "passed" if verification.valid else "failed",
                    "source_commit": route.source_commit,
                },
                ensure_ascii=False,
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )
        return paths

    def export_zip(self, route: OperatorRoute, output_zip: str | Path) -> Path:
        output_zip = Path(output_zip)
        tmp = output_zip.with_suffix("")
        if tmp.exists():
            shutil.rmtree(tmp)
        self.export_package(route, tmp)
        with zipfile.ZipFile(output_zip, "w", compression=zipfile.ZIP_DEFLATED) as archive:
            for path in sorted(tmp.rglob("*")):
                if path.is_file():
                    archive.write(path, path.relative_to(tmp.parent).as_posix())
        return output_zip
