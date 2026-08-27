from __future__ import annotations

import json
import math
import shutil
import zipfile
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass, field, replace
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
from fuzzyxai.core.alignment import AlignmentResult
from fuzzyxai.core.explain_plan import ExplainPlan
from fuzzyxai.core.risk_observer import observe_legacy_normalized_risk
from fuzzyxai.core.route import build_route
from fuzzyxai.core.types import AdaptedInput, OperatorRoute
from fuzzyxai.diagnostics.contracts import (
    BatchDiagnosticReport,
    DiagnosticReport,
    RepairExecutionContext,
)
from fuzzyxai.evidence import (
    KNOWN_ATTRIBUTION_CHANNELS,
    ExplanationClaim,
    ExplanationEdge,
    ExplanationEvidence,
    ExplanationGraph,
    ExplanationNode,
    HumanExplanation,
    TrainingRunAnalysis,
    build_attribution_map,
    build_class_concepts,
    build_explanation_claims,
    build_explanation_graph,
    build_object_trace,
    collect_data_evidence,
    collect_fuzzy_rule_activations,
    collect_model_internals,
    compose_human_explanation,
    detect_subgroup_averaging,
    determine_explanation_level,
    evaluate_explanation_quality,
    evaluate_explanation_quality_status,
    explanation_to_text,
    extract_rules,
    find_image_regions,
    find_similar_tabular_cases,
    find_tabular_counterfactuals,
    find_text_highlight_spans,
    is_image_like,
)
from fuzzyxai.explanation_quality import ExplanationQualityReport, build_quality_report
from fuzzyxai.operators import ReductionInput
from fuzzyxai.operators import compute_reduction as compute_operator_reduction
from fuzzyxai.planner import ExplanationPlanner
from fuzzyxai.proof.trace import build_proof_trace
from fuzzyxai.proof.verifier import VerificationResult, verify_proof_trace
from fuzzyxai.risk.risk_function import DEFAULT_RISK_WEIGHTS as DEFAULT_APPLICATION_RISK_WEIGHTS
from fuzzyxai.scientific_alignment import (
    AlignmentTransform,
    build_contribution_explanation_object,
    build_native_explanation_object,
    compute_real_alignment,
    compute_real_pre_interpretability,
)
from fuzzyxai.adapters.system_source import derive_system_source_evidence
from fuzzyxai.system_semantics import SystemEvidence, SystemObservation, build_system_evidence
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


def _extract_probability_vector(probabilities: Any) -> list[float] | None:
    """Pull out one object's real class-probability vector for an
    UncertaintyPolicy of method 'entropy'/'margin'. Handles a flat
    per-class vector (single object) or a batch (n_objects, n_classes),
    in which case the first object's row is used -- consistent with the
    rest of this Gamma/Delta/rho block's "first object" convention."""

    if probabilities is None:
        return None
    values = probabilities.tolist() if hasattr(probabilities, "tolist") else probabilities
    if not isinstance(values, (list, tuple)) or not values:
        return None
    first = values[0]
    if isinstance(first, (list, tuple)):
        values = first
    if not values or not all(isinstance(item, (int, float)) for item in values):
        return None
    return [float(item) for item in values]


def _normalized_entropy(probability_vector: list[float]) -> float | None:
    """Shannon entropy of a real class-probability vector, normalized to
    [0, 1] by the maximum possible entropy for that many classes."""

    total = sum(probability_vector)
    if total <= 0 or len(probability_vector) < 2:
        return None
    probabilities = [max(0.0, value) / total for value in probability_vector]
    entropy = -sum(p * math.log(p) for p in probabilities if p > 0)
    max_entropy = math.log(len(probability_vector))
    if max_entropy <= 0:
        return None
    return max(0.0, min(1.0, entropy / max_entropy))


def _predictive_margin_uncertainty(probability_vector: list[float]) -> float | None:
    """1 - (top probability - second probability): a confident, well-
    separated prediction has a wide margin and low uncertainty; a close
    call between the top two classes has a narrow margin and high
    uncertainty. Matches the margin computation already used by the
    dissertation's own chapter 5 reference demo (apps/chapter5_web_demo.py)."""

    if len(probability_vector) < 2:
        return None
    ranked = sorted(probability_vector, reverse=True)
    margin = ranked[0] - ranked[1]
    return max(0.0, min(1.0, 1.0 - margin))


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
        return render_visual_spec(self.visual_spec, view=selected_view, output_path=output, selector=self.selector)


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
    system_evidence: SystemEvidence | None = None

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
    def system(self) -> SystemEvidence | None:
        """Executed system-operator evidence, if this route declared one."""

        return self.system_evidence

    @property
    def claims(self) -> tuple[ExplanationClaim, ...]:
        values = self.view_model.claims
        return tuple(ExplanationClaim.from_dict(item) for item in values) if isinstance(values, (list, tuple)) else ()

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
    def required_missing_channels(self) -> tuple[str, ...]:
        return tuple(self.view_model.explanation_level.get("required_missing_channels", ()))

    @property
    def optional_missing_channels(self) -> tuple[str, ...]:
        return tuple(self.view_model.explanation_level.get("optional_missing_channels", ()))

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
            payload = cast("dict[str, Any]", self.view_model.to_dict(include_raw=include_raw))
            if self.system_evidence is not None:
                payload["system_evidence"] = self.system_evidence.audit_dict()
            return payload
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
            raise TypeError(f"unknown or unavailable audience profile: {audience}")
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

    def full_report(self, audience: str = "domain_user", *, level: str = "reader") -> str:
        """P16 section 20 / P17: a deterministic, 18-point full explanation
        report, in two levels.

        1. what was analyzed; 2. which model was used; 3. what prediction
        was obtained; 4. how this architecture actually reached it
        (linear reconstruction / tree path / ensemble votes / attribution
        map, whichever this model family produced); 5. supporting evidence;
        6. contradicting evidence; 7. how typical the object is for the
        data; 8. nearest examples and which are counterexamples; 9. what a
        counterfactual analysis would need to change; 10. training-time
        behavior; 11. agreement/conflict between explanation sources (Γ);
        12. whether simplification was performed and the resulting loss
        (Δ); 13. the five components ρ was built from; 14. the permitted
        action; 15. where every material statement comes from; 16. which
        required evidence is missing; 17. which channels do not apply to
        this architecture; 18. the explanation's limitations. Not an SLM —
        every line is built from evidence already collected during
        ``explain()``; a section is omitted (never fabricated) when its
        evidence is genuinely absent.

        ``level="reader"`` (the default): sections 5/6 show only the
        top-ranked 4-6 supporting / 3-4 contradicting factors (already
        ranked by ``rank_human_claims`` — no new ranking logic), each with
        its real computed chain (raw -> transformed -> coefficient ->
        contribution for a linear model, node-by-node for a tree, etc.),
        not a same-phrase list of every feature. ``level="audit"``: every
        supporting/contradicting claim, unabridged — the same content as
        before this parameter existed.
        """

        if level not in {"reader", "audit"}:
            raise ValueError("level must be 'reader' or 'audit'")
        he = self.explain_for(audience)
        lines: list[str] = []

        def section(number: int, title: str, body_lines: Sequence[str]) -> None:
            if not body_lines:
                return
            lines.append(f"## {number}. {title}")
            lines.extend(body_lines)
            lines.append("")

        trace = self.view_model.trace
        section(
            1,
            "Что анализировалось",
            [f"- Объект: {trace.get('object_ids', ['?'])[0] if trace.get('object_ids') else '?'}", f"- Версия данных: {trace.get('dataset_version', '?')}", f"- Входной hash: {trace.get('input_sha256', '?')}"],
        )
        section(
            2,
            "Какая модель использовалась",
            [f"- Модель: {trace.get('model_type', '?')} (адаптер {trace.get('adapter_id', '?')})", f"- Отпечаток модели (fingerprint): {trace.get('model_fingerprint', '?')}"],
        )

        # P18 item 7: never print the raw prediction array/label — reuse the
        # same domain-language decision text already built by
        # compose_human_explanation (he.decision), so a classifier's
        # prediction reads as its real class label, not "[1]"/"[0]".
        prediction = self.view_model.model
        prediction_lines = [f"- {he.decision.explanation}"]
        if isinstance(prediction.get("score"), (int, float)):
            prediction_lines.append(f"- Модельный балл: {round(float(prediction['score']), 4)}")
        section(3, "Какой прогноз получен", prediction_lines)

        mechanism_lines: list[str] = []
        for internals in self.view_model.layers.get("model_internals", []):
            if internals.get("linear_terms"):
                mechanism_lines.append(f"- Линейное разложение: {len(internals['linear_terms'])} слагаемых, восстановленная оценка {internals.get('reconstructed_score')}.")
            if internals.get("decision_path"):
                mechanism_lines.append(f"- Путь по дереву решений: {len(internals['decision_path'])} узлов, лист {internals.get('leaf_id')}.")
            if internals.get("ensemble_votes") is not None:
                mechanism_lines.append(f"- Голосование ансамбля: {len(internals['ensemble_votes'])} моделей, расхождение {internals.get('ensemble_disagreement')}.")
        for attribution in self.view_model.layers.get("attribution_maps", []):
            mechanism_lines.append(f"- Карта атрибуции ({attribution.get('method')}): диапазон [{attribution.get('min_value')}, {attribution.get('max_value')}].")
            completeness = attribution.get("completeness")
            if isinstance(completeness, Mapping) and completeness:
                if completeness.get("status") == "measured":
                    mechanism_lines.append(
                        "- IG completeness: "
                        f"target class={completeness.get('target_class')}; "
                        f"baseline={completeness.get('baseline')}; "
                        f"output space={completeness.get('output_space')}; "
                        f"absolute residual={completeness.get('completeness_residual')}; "
                        f"relative residual={completeness.get('completeness_relative_error')}. "
                        "Меньший residual означает более точное выполнение completeness identity в указанном output space."
                    )
                else:
                    mechanism_lines.append(
                        f"- IG completeness не оценена: {completeness.get('reason', 'причина не указана')}."
                    )
        if not mechanism_lines:
            contribution_method = self.view_model.model.get("contribution_method")
            mechanism_lines = [f"- Метод объяснения: {contribution_method}."] if contribution_method else []
        section(4, "Как именно эта архитектура сформировала данный прогноз", mechanism_lines)

        all_supports = list(he.details.supports)
        all_contradicts = list(he.details.contradicts)
        shown_supports = all_supports if level == "audit" else all_supports[:6]
        shown_contradicts = all_contradicts if level == "audit" else all_contradicts[:4]
        supports_lines = [f"- **{item.title}:** {item.explanation}" for item in shown_supports]
        if level == "reader" and len(all_supports) > len(shown_supports):
            supports_lines.append(f"- (показаны {len(shown_supports)} из {len(all_supports)}; полный список — full_report(level='audit'))")
        contradicts_lines = [f"- **{item.title}:** {item.explanation}" for item in shown_contradicts]
        if level == "reader" and len(all_contradicts) > len(shown_contradicts):
            contradicts_lines.append(f"- (показаны {len(shown_contradicts)} из {len(all_contradicts)}; полный список — full_report(level='audit'))")
        section(5, "Какие вычисленные сведения поддерживают результат", supports_lines)
        section(6, "Какие вычисленные сведения ему противоречат", contradicts_lines)

        data_lines: list[str] = []
        for item in self.view_model.layers.get("data", []):
            if item.get("anomaly_labels"):
                data_lines.append(f"- Отклонения от эталона: {', '.join(item['anomaly_labels'])}.")
            for warning in item.get("warnings", ()):
                data_lines.append(f"- {warning}")
        section(7, "Насколько объект похож на данные, использованные как эталон", data_lines)

        similar_lines = [f"- **{item.title}:** {item.explanation}" for item in he.details.similar_cases]
        counterexamples = [case for case in self.similar_cases if case.get("is_counterexample")]
        if counterexamples:
            similar_lines.append(f"- Из них контрпримеров (другой класс, чем прогноз): {len(counterexamples)} из {len(self.similar_cases)}.")
        section(8, "Какие ближайшие примеры найдены и какие из них являются контрпримерами", similar_lines)

        # P18 item 7: render the already domain-translated, fully Russian
        # `.explanation` text (built by human.py's _change_statement) rather
        # than re-deriving a raw summary line from `.direction`, which
        # carries the internal English "increase"/"decrease" token.
        section(
            9,
            "Что необходимо изменить для смены решения модели",
            [f"- **{change.title}:** {change.explanation}" for change in he.what_would_change_result],
        )
        section(10, "Что известно о поведении объекта и модели во время обучения", [f"- **{item.title}:** {item.explanation}" for item in he.details.training])

        disagreement = self.view_model.disagreement
        if disagreement.get("gamma") is not None:
            components_text = "; ".join(f"{name} = {value:.4f}" for name, value in disagreement.get("components", {}).items())
            gamma_max = disagreement.get("gamma_max")
            gamma_lines = [
                f"- Γ = {disagreement['gamma']:.4f} — измеренное рассогласование двух объяснительных объектов по компонентам {{{components_text}}} (веса заданы в ExplainPlan.beta).",
                *( [f"- d_L = {float(disagreement['components']['d_L']):.4f} измерено только как диагностическая компонента и не входит в aggregate Γ при текущем ExplainPlan.beta."] if "d_L" in disagreement.get("components", {}) else []),
                f"- Сертифицировано (Γ ≤ gamma_max={gamma_max}): {'да' if gamma_max is not None and disagreement['gamma'] <= gamma_max else 'нет'}.",
            ]
        else:
            gamma_lines = ["- Γ не измерено: для этого объекта доступен только один канал локального объяснения — сравнивать не с чем."]
        section(11, "Какие источники объяснения согласованы или конфликтуют (Γ)", gamma_lines)

        if disagreement.get("delta") is not None:
            if self.system_evidence is not None and self.system_evidence.reduction is not None:
                reduction = self.system_evidence.reduction
                delta_lines = [
                    f"- Δ = {reduction.delta:.6f} — измеренная D_F потеря представления в маршруте F_source → Pi → F_reduced → iota → reconstructed representation.",
                    f"- Исходный интервал: {list(reduction.source_interval)}; reduced scalar: {reduction.reduced_scalar}; reconstructed interval: {list(reduction.reconstructed_interval)}; D_F terms: {reduction.distance_terms}.",
                    *( ["- Редукция выполнена без измеренной потери для данного объекта."] if reduction.delta == 0 else []),
                ]
            else:
                delta_lines = [f"- Δ = {disagreement['delta']:.6f} — измеренная D_F потеря представления при реальной операции Pi и обратном вложении iota. r_delta = {disagreement.get('r_delta')}."]
        else:
            delta_lines = [f"- Упрощение не выполнялось: {'нет операции редукции для этого типа модели' if disagreement.get('reduction_status') == 'not_applied' else 'редукция не измерена'}."]
        section(12, "Выполнялось ли упрощение представления и какая потеря Δ возникла", delta_lines)

        risk = self.view_model.risk
        if risk.get("rho") is not None:
            components_text = "; ".join(f"{name} = {value:.4f}" for name, value in risk.get("components", {}).items())
            risk_lines = [f"- ρ = {risk['rho']:.4f}, из слагаемых: {{{components_text}}}."]
        elif risk.get("partial_risk_score") is not None:
            # P18 item 2: an incomplete interface still discloses the
            # number it computed, but never under the name "ρ" — a partial
            # weighted average over fewer terms than the schema expects is
            # not the same quantity as the real, complete risk score.
            components_text = "; ".join(f"{name} = {value:.4f}" for name, value in risk.get("partial_components", {}).items())
            missing_text = ", ".join(risk.get("missing_required_components", ()))
            risk_lines = [
                f"- ρ НЕ вычислено как полное значение: интерфейс риска неполон (не хватает: {missing_text}).",
                f"- Частичный (неполный) риск-score = {risk['partial_risk_score']:.4f}, из слагаемых: {{{components_text}}} — это НЕ официальное ρ, только для справки.",
            ]
        else:
            risk_lines = ["- ρ не вычислено: нет ни одного измеримого слагаемого риска для этого объекта."]
        section(13, "Из каких компонент получено ρ", risk_lines)

        section(14, "Какое действие разрешено", [f"- **{he.recommended_action.title}:** {he.recommended_action.explanation}"])

        provenance_lines = [f"- {item.title}: claims {', '.join(item.claim_refs)}; evidence {', '.join(item.evidence_refs)}" for item in (*he.main_reasons, *he.concerns)]
        section(15, "Откуда происходит каждое существенное утверждение", provenance_lines)

        # P17: missing_required (section 16), not_applicable (section 17),
        # and quality-metric non-evaluation (folded into section 18's
        # limitations, never labeled "required") are three distinct things
        # and must not be merged into one undifferentiated list — a quality
        # check nobody ran is a limitation of the explanation, not a
        # required channel the explanation plan structurally needs.
        explanation_level = self.view_model.explanation_level
        missing_lines = [f"- {name}" for name in explanation_level.get("required_missing_channels", ())]
        section(16, "Какие обязательные сведения отсутствуют", missing_lines)

        section(17, "Какие каналы неприменимы к данной архитектуре", [f"- {name}" for name in explanation_level.get("not_applicable_channels", ())])

        # `he.details.limitations` also carries every contradicting claim
        # (compose_human_explanation folds contradicts into concerns) —
        # already shown in section 6, so repeating them here (uncapped,
        # even in reader mode) would defeat the whole point of capping
        # section 6. Exclude anything already surfaced there.
        contradict_titles = {item.title for item in all_contradicts}
        genuine_limitations = [item for item in he.details.limitations if item.title not in contradict_titles]
        shown_limitations = genuine_limitations if level == "audit" else genuine_limitations[:5]
        limitation_lines = [f"- {item.title}: {item.explanation}" for item in shown_limitations]
        if level == "reader" and len(genuine_limitations) > len(shown_limitations):
            limitation_lines.append(f"- (показаны {len(shown_limitations)} из {len(genuine_limitations)}; полный список — full_report(level='audit'))")
        limitation_lines.extend(
            f"- Дополнительная проверка не выполнена — {name}: {status['reason']}"
            for name, status in self.view_model.quality_status.items()
            if status.get("status") == "not_evaluated"
        )
        limitation_lines.extend(
            f"- Необязательный канал не предоставлен: {name}."
            for name in explanation_level.get("optional_missing_channels", ())
        )
        section(18, "Какие ограничения имеет объяснение", limitation_lines)

        if self.system_evidence is not None:
            system = self.system_evidence
            gamma = system.alignment
            uncertainty = system.uncertainty
            reduction = system.reduction
            risk = system.risk
            source_provider = str((system.source_evidence.metadata or {}).get("provider", "registered system source"))
            section(19, "Системный операторный маршрут", [
                f"- E_model построен из фактического source evidence ({source_provider}); T_ij={system.alignment_transform.transform_id} применён к этому объекту.",
                f"- Γ = {float(gamma['gamma']):.6f}; компоненты: " + "; ".join(f"{name}={float(value):.6f}" for name, value in gamma['components'].items()),
                *( [f"- d_L={float(gamma['components']['d_L']):.6f} — diagnostic-only и не входит в aggregate Γ при текущем ExplainPlan.beta."] if "d_L" in gamma.get("components", {}) else []),
                f"- U_model={uncertainty.u_model}, U_rules={uncertainty.u_rules}, U_trace={uncertainty.u_trace}; u_M={uncertainty.u_m} ({uncertainty.status}).",
                (
                    f"- F_int → Pi → F0 → iota: source={list(reduction.source_interval)}, reduced={reduction.reduced_scalar}, reconstructed={list(reduction.reconstructed_interval)}, D_F/Δ={reduction.delta}; "
                    + ("редукция выполнена без измеренной потери для данного объекта." if reduction.delta == 0 else "измерена положительная потеря uncertainty representation.")
                    if reduction is not None
                    else "- Редукция не применялась по ExplainPlan; Delta=0 по явно объявленной not_applied semantics."
                ),
                f"- I_pre(E_pre)={system.i_pre:.6f}; ρ={risk.rho}; status={risk.status}; candidate={risk.candidate_action}; action={risk.action}.",
                f"- Причина кандидатного действия: {risk.candidate_action_reason}. Итоговая причина: {risk.final_action_reason}.",
                *( ["- По зарегистрированной демонстрационной политике ExplainPlan числовой риск находится ниже theta_1, поэтому кандидатное действие — accept. Это заключение относится к внутренней политике маршрута и не является доказательством предметной корректности или безопасности."] if risk.candidate_action == "accept" else []),
                *( [f"- Численная threshold policy дала candidate action={risk.candidate_action}, но critical_override=true имеет приоритет: final action=block."] if risk.critical_override else []),
            ])

        return "\n".join(lines).strip() + "\n"

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
        # An action is a provenance question: follow directed ancestors only
        # so independent nearby branches never masquerade as its cause. Other
        # inspection selectors retain their existing connected explanation
        # neighborhood for exploratory use.
        related_ids = set(anchors)
        frontier = set(anchors)
        while frontier:
            next_frontier: set[str] = set()
            for edge in edges:
                if prefix != "action" and edge.source in frontier and edge.target not in related_ids:
                    next_frontier.add(edge.target)
                if edge.target in frontier and edge.source not in related_ids:
                    next_frontier.add(edge.source)
            related_ids.update(next_frontier)
            frontier = next_frontier
        related_edges = [edge for edge in edges if edge.source in related_ids and edge.target in related_ids]
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

        payload = {
            "explanation_level": dict(self.view_model.explanation_level),
            "claims": [claim.to_dict() for claim in self.claims],
            "graph": dict(self.view_model.explanation_graph),
            "diagnostics": list(self.view_model.diagnostics),
            "action": self.action,
            "trace": dict(self.view_model.trace),
            "quality_metrics": dict(self.view_model.quality_metrics),
        }
        if self.system_evidence is not None:
            payload["system_evidence"] = self.system_evidence.audit_dict()
        return payload

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
        selector: str | None = None,
    ):
        """``selector`` (provenance view only): a claim_id ("C-002"), a
        node_id ("action"), or a node_type ("claim") to focus the rendered
        subgraph on — defaults to "action". The full graph remains
        available via ``result.audit()``/``to_dict(detail="audit")``."""

        selected_view = kind or view
        selected_output = output if output is not None else output_path
        if backend == "matplotlib":
            from fuzzyxai.visualization.matplotlib_renderer import render_visual_spec
        elif backend == "plotly":
            from fuzzyxai.visualization.plotly_renderer import render_visual_spec
        else:
            raise ValueError(f"unsupported visualization backend: {backend}")
        return render_visual_spec(self.view_model.visual_spec, view=selected_view, output_path=selected_output, selector=selector)

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


@dataclass(frozen=True)
class ObservationContext:
    """Bundles dataset/reference/training observation, registered once on ``wrap()``.

    P15.6: without this, a caller who wants a comprehensive result had to
    separately call ``observe_training()``, remember to pass its result back
    into ``explain_one(training_run=..., include_training_trace=True)``, and
    repeat ``reference_data``/``reference_labels`` on every call — with no
    single place that ties "the data", "the trained model", and "the final
    explanation" together. Registering one ``ObservationContext`` here makes
    every subsequent ``explain_one()``/``explain()`` call automatically pick
    up the reference corpus and training history, while the model-only path
    (``FuzzyXAI.wrap(model).explain_one(x)``, no context at all) is
    unchanged. Explicit per-call arguments (``reference_data=``,
    ``training_run=``, ...) still always win over what's registered here.

    ``training_run`` is produced by a *separate* call to
    ``FuzzyXAI.wrap(model).observe_training(history=...)`` — data observation
    and model explanation remain two distinct steps, exactly as before; this
    object is only the place their outputs are combined for reuse.
    """

    reference_data: Any | None = None
    reference_labels: Any | None = None
    reference_ids: list[str] | None = None
    training_run: TrainingRunAnalysis | None = None
    dataset_version: str | None = None
    run_parameters: Mapping[str, Any] = field(default_factory=dict)
    system_observation: SystemObservation | None = None


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
        observation_context: ObservationContext | None = None,
    ):
        self.plan = plan or ExplainPlan.default()
        self._model_adapter = model_adapter
        self._resolution_report = resolution_report
        self._observation_context = observation_context
        # A reference corpus registered once at wrap() time, used by default
        # for similar-case evidence on every explain_one() call so the
        # caller doesn't have to repeat reference_data/reference_labels on
        # every call. Still overridable per-call. Explicit kwargs here win
        # over an ObservationContext's own reference_data, if both are given.
        context = observation_context
        self._reference_data = reference_data if reference_data is not None else (context.reference_data if context else None)
        self._reference_labels = reference_labels if reference_labels is not None else (context.reference_labels if context else None)
        self._reference_ids = reference_ids if reference_ids is not None else (context.reference_ids if context else None)

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
        observation_context: ObservationContext | None = None,
    ) -> FuzzyXAI:
        """Wrap a supported model using capability-based adapter resolution.

        ``reference_data``/``reference_labels``/``reference_ids`` register a
        reference corpus once, here, instead of on every ``explain_one()``
        call — when present, similar-case evidence is produced by default
        (see ``explain()``'s ``include_similar_cases``).

        ``observation_context`` (P15.6) is the comprehensive alternative:
        one ``ObservationContext`` bundling the reference corpus *and* a
        prior ``observe_training()`` result *and* dataset/run metadata, all
        auto-applied to every subsequent ``explain_one()``/``explain()``
        call. Explicit ``reference_data=`` etc. passed directly here still
        wins over the same field on ``observation_context``.
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
            observation_context=observation_context,
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
        include_training_trace: bool | None = None,
        include_model_knowledge: bool = True,
        additional_evidence: ExplanationEvidence | None = None,
        dataset_version: str | None = None,
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
        # P15.6: a registered ObservationContext auto-applies its training
        # history / dataset metadata too, the same "explicit call wins"
        # priority as the reference corpus above.
        context = self._observation_context
        if context is not None:
            if training_run is None and context.training_run is not None:
                training_run = context.training_run
                if include_training_trace is None:
                    include_training_trace = True
            if dataset_version is None and context.dataset_version is not None:
                dataset_version = context.dataset_version
            if run_parameters is None and context.run_parameters:
                run_parameters = context.run_parameters
        if include_training_trace is None:
            include_training_trace = False
        if dataset_version is None:
            dataset_version = "unversioned"
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

        # P16/P17: scientific Γ/Δ/ρ, replacing P15.7's heuristic proxies
        # (percent-of-claims / surrogate-fidelity-gap) with the
        # dissertation's real chapter-2/3 machinery: semantic_disagreement
        # over genuine ExplanationObject pairs for Γ, and
        # compute_interpretability_index (I_pre = exp(-L(E))) for the
        # interpretability term — see fuzzyxai.scientific_alignment.
        # Manual `evidence={"alignment"/"reduction"/"risk": ...}` still
        # always wins. Most single-channel sklearn explanations genuinely
        # have no second explanatory object to compare against, so Γ
        # honestly stays unmeasured for them — only models that supply a
        # real second channel (e.g. a fuzzy/ANFIS model's native rule
        # activations alongside its numeric contributions) get an automatic
        # Γ. This is intentionally more conservative than P15.7.
        #
        # P17: Δ is NOT derived automatically from `reconstruction_error`
        # (that is a reconstruction-fidelity check on the linear formula —
        # "does x.w+b match decision_function?" — not a measurement of
        # information lost by a real representation-reduction operation Π).
        # `reconstruction_error` stays a standalone quality metric
        # (evidence/metrics.py); Δ/reduction here is populated ONLY from
        # manually supplied `evidence={"reduction": ...}` — i.e. only when
        # a real reduction actually happened and was measured by the
        # caller. No automatic Π exists in this runtime for any sklearn
        # model family, so Δ correctly stays `not_applied` automatically.
        missing: list[str] = []
        required_missing: list[str] = []
        model_fingerprint = self._model_adapter.model_fingerprint()
        contributions_for_operators = dict(evidence.get("contributions", internal_evidence.get("contributions", {})))
        raw_activated_rules_for_operators = evidence.get("activated_rules", internal_evidence.get("activated_rules"))
        operator_source = str(internal_evidence.get("contribution_method") or prediction.adapter_id)
        native_explanation_object = (
            build_native_explanation_object(
                raw_activated_rules_for_operators,
                object_id=ids[0],
                model_fingerprint=model_fingerprint,
                source=operator_source,
            )
            if isinstance(raw_activated_rules_for_operators, Sequence)
            else None
        )
        derived_explanation_object = build_contribution_explanation_object(
            contributions_for_operators,
            object_id=ids[0],
            model_fingerprint=model_fingerprint,
            source=operator_source,
            score=score,
        )
        best_explanation_object = derived_explanation_object or native_explanation_object
        pre_interpretability = compute_real_pre_interpretability(best_explanation_object, plan=self.plan)

        alignment_data = evidence.get("alignment")
        alignment = None
        alignment_components_used: dict[str, float] = {}
        raw_transform = (
            alignment_data.get("transform") if isinstance(alignment_data, Mapping) else None
        ) or self.plan.alignment_policy.transform
        transform = None
        if isinstance(raw_transform, Mapping):
            try:
                transform = AlignmentTransform.from_dict(raw_transform)
            except (KeyError, TypeError, ValueError):
                transform = None
        real_alignment = compute_real_alignment(
            native_explanation_object,
            derived_explanation_object,
            plan=self.plan,
            transform=transform,
        )
        if real_alignment["gamma"] is not None:
            alignment_components_used = dict(real_alignment["components"])
            alignment = AlignmentResult(
                gamma=real_alignment["gamma"], gamma_max=real_alignment["gamma_max"],
                delta_t=0.0, certified=bool(real_alignment["certified"]),
            )
        # P18 item 1: alignment is only ever "expected" for this object when
        # the resolved ExplainPlan genuinely declares a second explanatory
        # channel (AlignmentPolicy.applicable) or the caller manually
        # measured one — never universally. An ordinary single-channel
        # model (linear/tree/ensemble) is then honestly `not_applicable`,
        # not `missing_required`, and is never penalized for lacking an
        # operation its own plan never calls for.
        alignment_expected = bool(self.plan.alignment_policy.applicable) or isinstance(alignment_data, Mapping)
        if alignment is not None:
            if not alignment.certified:
                diagnostics.append(
                    {
                        "code": "D_ij_alignment",
                        "reason": "alignment exceeds the configured gamma or transition-loss boundary",
                        "severity": "error",
                    }
                )
        elif alignment_expected:
            diagnostics.append(
                {
                    "code": "D_k_alignment_missing",
                    "reason": "ExplainPlan.alignment_policy declares alignment applicable but no second explanatory channel or manual evidence was available",
                    "severity": "warning",
                }
            )
            required_missing.append("alignment_required_channel")
        else:
            diagnostics.append(
                {
                    "code": "D_k_alignment_not_applicable",
                    "reason": "ExplainPlan.alignment_policy does not declare alignment applicable for this scenario (single explanatory channel)",
                    "severity": "info",
                }
            )

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
        # P18 item 1: same context-dependent requiredness for reduction
        # (Delta) — required only when ExplainPlan.reduction_policy declares
        # a real reduction operation (Pi) is part of this scenario, or the
        # caller measured one manually. This runtime has no automatic Pi for
        # any sklearn model family, so most scenarios correctly stay
        # `not_applicable`, never `missing_required`.
        reduction_expected = bool(self.plan.reduction_policy.applicable) or isinstance(reduction_data, Mapping)
        if reduction is not None:
            if not reduction.allowed:
                diagnostics.append(
                    {
                        "code": "D_reduction",
                        "reason": "representation reduction exceeds delta_max",
                        "severity": "error",
                    }
                )
        elif reduction_expected:
            diagnostics.append(
                {
                    "code": "D_k_reduction_missing",
                    "reason": "ExplainPlan.reduction_policy declares reduction applicable but no real reduction (Pi) was measured for this object",
                    "severity": "warning",
                }
            )
            required_missing.append("reduction_required_measurement")
        else:
            diagnostics.append(
                {
                    "code": "D_k_reduction_not_applicable",
                    "reason": "ExplainPlan.reduction_policy declares no reduction operation (Pi) for this scenario",
                    "severity": "info",
                }
            )

        structural_failure = (alignment is not None and not alignment.certified) or (reduction is not None and not reduction.allowed)

        risk_data = evidence.get("risk")
        risk = None
        risk_components_used: dict[str, float] = {}
        risk_thresholds = {
            "theta_1": self.plan.rho_accept,
            "theta_2": self.plan.rho_warning,
            "theta_3": self.plan.rho_audit,
            "theta_4": self.plan.rho_critical,
        }
        risk_status = "complete"
        missing_required_risk_components: list[str] = []
        risk_weights_base: dict[str, float] = {}
        partial_risk_score: float | None = None
        if isinstance(risk_data, Mapping):
            risk_components_used = dict(risk_data.get("components", {}))
            risk_weights_base = dict(risk_data["weights"])
            thresholds = dict(risk_data.get("thresholds", risk_thresholds))
            risk = observe_legacy_normalized_risk(
                dict(risk_data["components"]),
                dict(risk_data["weights"]),
                thresholds,
                int(risk_data.get("chi_r_crit", 0)),
            )
        else:
            # Canonical 5-component chapter-3 risk formula:
            # rho = w_p*predicted_risk + w_u*uncertainty
            #     + w_I*interpretability_gap + w_Delta*reduction_loss
            #     + w_R*diagnostic (see fuzzyxai.risk.risk_function
            #     .DEFAULT_RISK_WEIGHTS, overridable via
            #     ExplainPlan.metadata['risk_weights']).
            #
            # P18 items 1/3: which of the 5 keys are even *expected* for
            # this object is itself context-dependent, not universal.
            # "uncertainty" is expected only when ExplainPlan.uncertainty_policy
            # declares a real source (never surrogate_fidelity_gap — that
            # measures explanation fidelity, not predictive uncertainty);
            # "reduction_loss" is expected only when reduction is genuinely
            # part of this scenario (reduction_expected, computed above).
            # A component that was never expected is not_applicable and
            # simply excluded from the schema — never `missing_required`.
            risk_weights_full = dict(self.plan.metadata.get("risk_weights", DEFAULT_APPLICATION_RISK_WEIGHTS))
            uncertainty_policy = self.plan.uncertainty_policy
            expected_component_names = {"predicted_risk", "interpretability_gap", "diagnostic"}
            if uncertainty_policy.applicable:
                expected_component_names.add("uncertainty")
            if reduction_expected:
                expected_component_names.add("reduction_loss")
            risk_weights_base = {key: risk_weights_full[key] for key in expected_component_names if key in risk_weights_full}

            candidate_risk_components: dict[str, float] = {}
            if isinstance(score, (int, float)):
                candidate_risk_components["predicted_risk"] = max(0.0, min(1.0, 1.0 - float(score)))
            uncertainty_value: float | None = None
            if uncertainty_policy.method == "ensemble_disagreement":
                ensemble_disagreement_value = internal_evidence.get("ensemble_disagreement")
                if isinstance(ensemble_disagreement_value, (int, float)):
                    uncertainty_value = max(0.0, min(1.0, float(ensemble_disagreement_value)))
            elif uncertainty_policy.method in {"entropy", "margin"}:
                probability_vector = _extract_probability_vector(prediction.probabilities)
                if probability_vector is not None:
                    uncertainty_value = _normalized_entropy(probability_vector) if uncertainty_policy.method == "entropy" else _predictive_margin_uncertainty(probability_vector)
            elif uncertainty_policy.method == "calibrated_interval":
                interval_width = internal_evidence.get("calibrated_interval_width")
                if isinstance(interval_width, (int, float)):
                    uncertainty_value = max(0.0, min(1.0, float(interval_width)))
            if uncertainty_value is not None:
                candidate_risk_components["uncertainty"] = uncertainty_value
            if pre_interpretability["i_pre"] is not None:
                candidate_risk_components["interpretability_gap"] = max(0.0, min(1.0, 1.0 - float(pre_interpretability["i_pre"])))
            if reduction is not None:
                candidate_risk_components["reduction_loss"] = max(0.0, min(1.0, float(reduction.delta)))
            substantive_components = {key: value for key, value in candidate_risk_components.items() if key != "diagnostic" and key in risk_weights_base}
            if substantive_components:
                candidate_risk_components["diagnostic"] = 1.0 if structural_failure else 0.0
                risk_components = {key: value for key, value in candidate_risk_components.items() if key in risk_weights_base}
                # P17/P18: components genuinely *expected* by this object's
                # resolved schema but still lacking a real value are NOT
                # silently dropped-and-renormalized-away — the interface is
                # marked incomplete, disclosed, and prevented from reading
                # as a plain "accept" below (see required_missing). A
                # component that was never expected in the first place
                # (excluded from risk_weights_base above) never reaches
                # this check at all.
                missing_required_risk_components = sorted(set(risk_weights_base) - set(risk_components))
                if missing_required_risk_components:
                    risk_status = "incomplete"
                risk_components_used = risk_components
                risk = observe_legacy_normalized_risk(
                    risk_components,
                    {key: risk_weights_base[key] for key in risk_components},
                    risk_thresholds,
                    1 if structural_failure else 0,
                )
        # A legacy subset is useful only as a partial score.  Its implicit
        # normalization must never be exported under the dissertation name ρ.
        if risk is not None and not isinstance(risk_data, Mapping) and abs(sum(risk_weights_base.values()) - 1.0) > 1e-9:
            risk_status = "incomplete"
            partial_risk_score = risk.rho
            missing_required_risk_components = sorted(set(DEFAULT_APPLICATION_RISK_WEIGHTS) - set(risk_components_used))
        if risk is not None:
            if risk.chi_r_crit:
                diagnostics.append(
                    {
                        "code": "D_risk_critical",
                        "reason": "critical risk rupture forbids automatic acceptance",
                        "severity": "critical",
                    }
                )
            if risk_status == "incomplete":
                diagnostics.append(
                    {
                        "code": "D_risk_incomplete_interface",
                        "reason": f"risk components expected by this object's schema but not measured: {missing_required_risk_components}",
                        "severity": "warning",
                    }
                )
                required_missing.append("risk_required_components")
                # P18 item 2: an incomplete interface never surfaces its
                # renormalized weighted average as if it were the real,
                # complete rho — that number is disclosed separately as
                # partial_risk_score (see the risk dict below), never as
                # `rho` itself.
                partial_risk_score = risk.rho
        else:
            diagnostics.append({"code": "D_k_risk_missing", "reason": "risk evidence was not supplied", "severity": "warning"})

        # P18 item 2: rho is a real, complete number only when every
        # component this object's resolved schema expects is genuinely
        # present. A critical structural rupture still forces "block"
        # even under an incomplete interface (chi_r_crit is never partial).
        if risk is not None and risk.chi_r_crit or risk is not None and risk_status == "complete":
            action = risk.action
        else:
            action = "review"
        if structural_failure and action != "block":
            action = "review"
        # P19: a declared system route is executed here, before graph/report
        # projection.  It is never reconstructed by a renderer or exporter.
        system_evidence: SystemEvidence | None = None
        if context is not None and context.system_observation is not None:
            try:
                registered_transform = AlignmentTransform.from_dict(self.plan.alignment_policy.transform)
                system_source = derive_system_source_evidence(
                    object_id=ids[0], model_fingerprint=model_fingerprint,
                    prediction=prediction, internal_evidence=internal_evidence,
                    source_interface_id=registered_transform.source_interface,
                    risk_class=context.system_observation.risk_class,
                    trace=context.system_observation.trace,
                    model_trace=context.system_observation.model_trace,
                    source_refs=context.system_observation.source_refs,
                )
                system_evidence = build_system_evidence(
                    object_id=ids[0], model_fingerprint=model_fingerprint, source=system_source, plan=self.plan,
                    observation=context.system_observation,
                )
            except ValueError as exc:
                diagnostics.append({"code": "D_system_route_incomplete", "reason": str(exc), "severity": "warning"})
                required_missing.append("system_operator_route")
            else:
                diagnostics.extend(system_evidence.diagnostics)
                action = system_evidence.risk.action
                risk_status = system_evidence.risk.status
                # The declared system route supplied the required T_ij and
                # Π evidence; generic single-channel placeholders above no
                # longer govern this result's action.
                required_missing = [item for item in required_missing if item not in {
                    "alignment_required_channel", "reduction_required_measurement", "risk_required_components",
                }]
                diagnostics = [item for item in diagnostics if item.get("code") not in {
                    "D_k_alignment_not_applicable", "D_k_alignment_missing",
                    "D_k_reduction_not_applicable", "D_k_reduction_missing",
                    "D_risk_incomplete_interface", "D_k_risk_missing",
                }]
                risk_components_used = {
                    key: float(value) for key, value in system_evidence.risk.components.items()
                    if isinstance(value, (int, float))
                }
                risk_weights_base = dict(system_evidence.risk.weights)
                risk_thresholds = dict(self.plan.metadata.get("system_risk_thresholds", {
                    "theta_1": self.plan.rho_accept, "theta_2": self.plan.rho_warning,
                    "theta_3": self.plan.rho_audit, "theta_4": self.plan.rho_critical,
                }))
        route = [
            {"id": "model", "label": "Model", "status": "passed"},
            {"id": "adapter", "label": prediction.adapter_id, "status": "passed"},
            {"id": "alignment", "label": "T_ij", "status": "passed" if system_evidence is not None else ("passed" if (alignment and alignment.certified) else ("warning" if alignment_expected else "not_applied"))},
            {"id": "reduction", "label": "Delta", "status": (system_evidence.reduction_status if system_evidence is not None else ("passed" if (reduction and reduction.allowed) else ("warning" if reduction_expected else "not_applied")))},
            {"id": "risk", "label": "Risk", "status": "blocked" if action == "block" else ("passed" if (system_evidence is not None or (risk and risk_status == "complete")) else "warning")},
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
                    query_row=rows[0],
                )
        contributions = dict(evidence.get("contributions", internal_evidence.get("contributions", {})))
        contribution_method = evidence.get("contribution_method", internal_evidence.get("contribution_method"))
        raw_activated_rules = evidence.get("activated_rules", internal_evidence.get("activated_rules"))
        fuzzy_rule_activations = (
            collect_fuzzy_rule_activations(raw_activated_rules, object_id=ids[0])
            if isinstance(raw_activated_rules, Sequence) and raw_activated_rules
            else []
        )
        model_internals_evidence = collect_model_internals(
            internal_evidence,
            object_id=ids[0],
            model_family=getattr(self._model_adapter, "model_family", type(self._model_adapter).__name__),
        )
        model_internals = [model_internals_evidence] if model_internals_evidence is not None else []
        if not model_internals:
            missing.append("model_internals")
        if include_model_knowledge and not rules and not fuzzy_rule_activations and not concepts:
            # P18 item 12: rule-based knowledge is genuinely absent only
            # when none of the three channels that can supply it produced
            # anything — a model with a real reference corpus (concepts
            # built from it) must not also be told "model rules are
            # missing" in the same result.
            missing.append("model_rules_or_concepts")
        text_highlights = []
        image_representations = []
        attribution_maps = []
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
                attribution_channel = next((key for key in KNOWN_ATTRIBUTION_CHANNELS if internal_evidence.get(key) is not None), None)
                if attribution_channel is not None:
                    predicted_target = prediction.predictions[0] if isinstance(prediction.predictions, (list, tuple)) and prediction.predictions else prediction.predictions
                    attribution_maps = [
                        build_attribution_map(
                            raw_object,
                            internal_evidence[attribution_channel],
                            object_id=ids[0],
                            method=attribution_channel,
                            target=str(predicted_target) if predicted_target is not None else None,
                            baseline="zero baseline" if attribution_channel == "integrated_gradients" else "unspecified baseline",
                            completeness_error=internal_evidence.get("completeness_error"),
                            completeness=internal_evidence.get("ig_completeness"),
                        )
                    ]
                else:
                    missing.append("attribution_map")
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
                # find_similar_tabular_cases is a pure similarity search — it
                # doesn't know this object's own prediction, so it can never
                # set is_counterexample itself. Without this, a nearest
                # neighbor with a *different* label was rendered as
                # "additional confirmation" instead of a counterexample.
                raw_predictions = prediction.predictions
                predicted_for_similarity = raw_predictions[0] if isinstance(raw_predictions, (list, tuple)) and raw_predictions else raw_predictions
                similar_cases = [
                    replace(case, is_counterexample=True)
                    if case.reference_label is not None and str(case.reference_label) != str(predicted_for_similarity)
                    else case
                    for case in similar_cases
                ]
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
            attribution_maps=[*attribution_maps, *additional.attribution_maps],
            fuzzy_rule_activations=[*fuzzy_rule_activations, *additional.fuzzy_rule_activations],
            model_internals=[*model_internals, *additional.model_internals],
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
            provenance={
                "dataset_version": dataset_version,
                "model_fingerprint": self._model_adapter.model_fingerprint(),
                "model_type": prediction.model_type,
                "adapter_id": prediction.adapter_id,
                "training_run": dict(training_run.provenance) if training_run is not None else None,
                "split": (run_parameters or {}).get("split"),
            },
            system_evidence=system_evidence.audit_dict() if system_evidence is not None else None,
        )
        explanation_level = determine_explanation_level(
            explanation_evidence,
            contribution_method=str(contribution_method) if contribution_method and contributions else None,
            operator_channels={
                "alignment": system_evidence is not None or alignment is not None,
                "reduction": (system_evidence is not None and system_evidence.reduction is not None) or reduction is not None,
                "risk": system_evidence is not None or risk is not None,
            },
            # default True (unchanged behavior) when the adapter doesn't
            # declare the capability either way — only a model that
            # explicitly says it has no native rules / no local
            # contributions gets those channels reported as not_applicable
            # rather than missing.
            native_rules_supported=bool(adapter_capabilities.get("rules", True)),
            local_contributions_supported=bool(adapter_capabilities.get("local_contributions", True)),
            alignment_applicable=alignment_expected,
            reduction_applicable=reduction_expected,
            required_channels=tuple({
                "prediction", "call_trace", "data_profile", "risk",
                *({"training_history"} if include_training_trace else set()),
                *({"similar_cases"} if include_similar_cases else set()),
                *({"counterfactuals"} if include_counterfactuals else set()),
                *({"alignment"} if alignment_expected else set()),
                *({"reduction"} if reduction_expected else set()),
            }),
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
        quality_supplied_metrics = {
            **dict(evidence.get("quality_metrics", {})),
            **({"fidelity": float(surrogate_fidelity)} if surrogate_fidelity is not None else {}),
            **(
                {"reconstruction_error": float(internal_evidence["reconstruction_error"])}
                if internal_evidence.get("reconstruction_error") is not None
                else {}
            ),
        }
        quality_metrics = evaluate_explanation_quality(
            explanation_evidence,
            graph,
            contributions=contributions,
            supplied_metrics=quality_supplied_metrics,
        )
        quality_status = evaluate_explanation_quality_status(
            explanation_evidence,
            graph,
            contributions=contributions,
            supplied_metrics=quality_supplied_metrics,
        )
        visual_spec = build_visual_spec(
            explanation_evidence,
            claims,
            graph,
            prediction=prediction_payload,
            action=action,
            contributions=contributions,
            explanation_level=explanation_level.to_dict(),
            domain_language=self.plan.domain_language,
        )
        view_model = ExplanationViewModel(
            model={
                **prediction.to_dict(),
                "score": score,
                "contributions": contributions,
                "contribution_method": contribution_method,
                "contribution_limitations": list(internal_evidence.get("limitations", [])),
            },
            fuzzy={
                "memberships": dict(evidence.get("memberships", {})),
                # P16 section 17: whatever membership-function policies are
                # registered on this ExplainPlan — parameters, origin,
                # version — are always disclosed alongside any membership
                # evidence, so no membership function in the output is an
                # "unexplained triangle". Empty when none are registered.
                "membership_policies": {name: policy.to_dict() for name, policy in self.plan.membership_policies.items()},
            },
            route=route,
            disagreement={
                "components": dict(system_evidence.alignment["components"]) if system_evidence is not None else dict(alignment_components_used),
                "gamma": system_evidence.alignment["gamma"] if system_evidence is not None else (alignment.gamma if alignment else None),
                "gamma_max": system_evidence.alignment["gamma_max"] if system_evidence is not None else (alignment.gamma_max if alignment else None),
                "delta_t": alignment.delta_t if alignment else None,
                "alignment_status": "measured" if system_evidence is not None else ("measured" if alignment is not None else ("missing" if alignment_expected else "not_applied")),
                "delta": system_evidence.reduction.delta if system_evidence is not None and system_evidence.reduction is not None else (reduction.delta if reduction else None),
                "r_delta": reduction.r_delta if reduction else None,
                "reduction_status": system_evidence.reduction_status if system_evidence is not None else ("measured" if reduction is not None else ("missing" if reduction_expected else "not_applied")),
                "pre_interpretability": system_evidence.i_pre if system_evidence is not None else pre_interpretability["i_pre"],
                "pre_interpretability_status": "measured" if system_evidence is not None else pre_interpretability["status"],
                # P18 item 4: the weights/thresholds that would govern
                # alignment/reduction under this ExplainPlan, disclosed
                # even when neither operator was applicable for this object.
                "beta": dict(self.plan.beta),
                "gamma_critical": self.plan.gamma_critical,
                "gamma_warning": self.plan.gamma_warning,
                "delta_critical": self.plan.delta_critical,
                "delta_warning": self.plan.delta_warning,
                "alignment_policy": self.plan.alignment_policy.to_dict(),
                "alignment_transform": system_evidence.alignment_transform.to_dict() if system_evidence is not None else (real_alignment.get("transform") if isinstance(real_alignment, Mapping) else None),
                "reduction_policy": self.plan.reduction_policy.to_dict(),
            },
            risk={
                "components": dict(system_evidence.risk.components) if system_evidence is not None else dict(risk_components_used),
                # P18 item 2: rho is disclosed as a real, complete number
                # only when risk_status == "complete" -- an incomplete
                # interface never surfaces a renormalized average under the
                # name "rho". The same number is still disclosed, honestly
                # labeled, as partial_risk_score/partial_components.
                "rho": system_evidence.risk.rho if system_evidence is not None else (risk.rho if (risk is not None and risk_status == "complete") else None),
                "partial_risk_score": system_evidence.risk.partial_risk_score if system_evidence is not None else partial_risk_score,
                "partial_components": dict(risk_components_used) if (risk is not None and risk_status == "incomplete") else {},
                "chi_r_crit": int(system_evidence.risk.critical_override) if system_evidence is not None else (risk.chi_r_crit if risk else None),
                "critical_override": bool(system_evidence is not None and system_evidence.risk.critical_override),
                "action": action,
                "status": system_evidence.risk.status if system_evidence is not None else (risk_status if risk is not None else "missing"),
                "missing_required_components": (
                    [
                        key for key, value in system_evidence.risk.components.items()
                        if value is None and system_evidence.risk.weights.get(key, 0.0) > 0.0
                    ]
                    if system_evidence is not None
                    else list(missing_required_risk_components)
                ),
                # P18 item 4: the parameters that actually determined this
                # result, serialized so the audit output is self-contained.
                "risk_weights": dict(system_evidence.risk.weights) if system_evidence is not None else dict(risk_weights_base),
                "risk_thresholds": dict(risk_thresholds),
                "uncertainty_policy": self.plan.uncertainty_policy.to_dict(),
                "reduction_policy": self.plan.reduction_policy.to_dict(),
                "alignment_policy": self.plan.alignment_policy.to_dict(),
                "alignment_transform": system_evidence.alignment_transform.to_dict() if system_evidence is not None else (real_alignment.get("transform") if isinstance(real_alignment, Mapping) else None),
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
            quality_status=quality_status,
            explanation_level=explanation_level.to_dict(),
            visual_spec=visual_spec.to_dict(),
        )
        return ModelExplanationResult(prediction=prediction, view_model=view_model, system_evidence=system_evidence)

    def explain_one(
        self,
        input_object: Any,
        *,
        object_id: str = "object_0",
        include_similar_cases: bool | None = None,
        include_counterfactuals: bool = False,
        include_training_trace: bool | None = None,
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
        training_run_id: str | None = None,
        training_method: str | None = None,
        epoch_source: str | None = None,
        final_checkpoint_ref: str | None = None,
    ) -> TrainingRunAnalysis:
        """Create auditable object trajectories and subgroup diagnostics.

        Only ``history`` (per-object per-epoch metrics, already computed by
        the caller) is actually used. ``train_data``/``val_data``/
        ``checkpoints`` are accepted but not processed — no raw dataset or
        checkpoint inspection is performed here — this is disclosed via
        ``TrainingRunAnalysis.limitations`` rather than silently ignored.
        """

        limitations: list[str] = []
        if train_data is not None or val_data is not None:
            limitations.append("train_data/val_data were supplied but are not inspected; only the precomputed history mapping is used.")
        if checkpoints is not None:
            limitations.append("checkpoints were supplied but are not inspected; only the precomputed history mapping is used.")
        object_history = history.get("objects", {})
        fingerprint = self._model_adapter.model_fingerprint() if self._model_adapter is not None else None
        provenance = {
            "training_run_id": training_run_id, "model_fingerprint": fingerprint,
            "final_model_fingerprint": fingerprint, "training_method": training_method,
            "epoch_source": epoch_source, "final_checkpoint_ref": final_checkpoint_ref,
            "number_of_epochs": max((len(values) for values in object_history.values()), default=0),
        }
        traces = {str(object_id): build_object_trace(str(object_id), metrics, provenance=provenance) for object_id, metrics in object_history.items()}
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
        return TrainingRunAnalysis(traces=traces, subgroups=subgroups, rules=rules, limitations=tuple(limitations), provenance=provenance)

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
