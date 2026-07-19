from __future__ import annotations

import json
import shutil
import zipfile
from dataclasses import dataclass
from datetime import datetime, timezone
from html import escape
from hashlib import sha256
from pathlib import Path
from typing import Any, Mapping, Sequence

from fuzzyxai.adapters.base import BaseAdapter
from fuzzyxai.adapters.model import ModelAdapter, ModelPrediction, resolve_model_adapter
from fuzzyxai.core.explain_plan import ExplainPlan
from fuzzyxai.core.types import AdaptedInput, OperatorRoute
from fuzzyxai.operators import AlignmentInput, ReductionInput, RiskInput
from fuzzyxai.operators import compute_alignment as compute_operator_alignment
from fuzzyxai.operators import compute_reduction as compute_operator_reduction
from fuzzyxai.operators import observe_risk as observe_operator_risk
from fuzzyxai.proof.trace import build_proof_trace
from fuzzyxai.proof.verifier import VerificationResult, verify_proof_trace
from fuzzyxai.viz import save_proof_trace_json, save_route_json, write_traceability_artifacts
from fuzzyxai.viz.matplotlib_dashboard import render_dashboard
from fuzzyxai.core.route import build_route
from fuzzyxai.evidence import (
    ExplanationEvidence,
    TrainingRunAnalysis,
    build_class_concepts,
    build_explanation_graph,
    build_explanation_claims,
    build_object_trace,
    collect_data_evidence,
    compose_human_explanation,
    detect_subgroup_averaging,
    determine_explanation_level,
    explanation_to_text,
    evaluate_explanation_quality,
    extract_rules,
    find_similar_tabular_cases,
    find_tabular_counterfactuals,
)
from fuzzyxai.visualization.spec import build_visual_spec
from fuzzyxai.visualization.view_model import ExplanationViewModel


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
class ExplanationInspection:
    """Focused, serializable view of one claim, rule, or evidence node."""

    selector: str
    target: Mapping[str, Any]
    related_claims: Sequence[Mapping[str, Any]]
    related_nodes: Sequence[Mapping[str, Any]]
    related_edges: Sequence[Mapping[str, Any]]
    visual_spec: Mapping[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return {
            "selector": self.selector,
            "target": dict(self.target),
            "related_claims": [dict(item) for item in self.related_claims],
            "provenance": self.provenance(),
        }

    def provenance(self) -> dict[str, Any]:
        return {
            "nodes": [dict(item) for item in self.related_nodes],
            "edges": [dict(item) for item in self.related_edges],
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


@dataclass(frozen=True)
class ModelExplanationResult:
    """Public result returned by ``FuzzyXAI.wrap(...).explain(...)``."""

    prediction: ModelPrediction
    view_model: ExplanationViewModel

    @property
    def action(self) -> str:
        return str(self.view_model.risk.get("action", "review"))

    @property
    def claims(self) -> tuple[Mapping[str, Any], ...]:
        values = self.view_model.claims
        return tuple(values) if isinstance(values, (list, tuple)) else tuple()

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

    def to_dict(self) -> dict[str, Any]:
        return self.view_model.to_dict()

    def export_json(self, path: str | Path) -> Path:
        return self.view_model.export_json(path)

    @property
    def explanation_graph(self) -> Mapping[str, Any]:
        return self.view_model.explanation_graph

    def summary(self, level: str = "user") -> str:
        """Return evidence-backed user, expert, or audit text."""

        payload = self.view_model.human_explanations.get(level)
        if not isinstance(payload, Mapping):
            raise ValueError(f"unknown or unavailable explanation level: {level}")
        from fuzzyxai.evidence.contracts import HumanExplanation

        return explanation_to_text(HumanExplanation(**payload))

    def overview(self) -> str:
        """Answer the five operational questions using claim-grounded text."""

        groups: dict[str, list[Mapping[str, Any]]] = {}
        for claim in self.claims:
            groups.setdefault(str(claim.get("claim_type")), []).append(claim)

        def line(title: str, candidates: Sequence[str], fallback: str) -> str:
            selected = [claim for kind in candidates for claim in groups.get(kind, [])]
            if not selected:
                return f"**{title}.** {fallback}"
            claim = selected[0]
            return f"**{title}.** {claim.get('statement')} [{claim.get('claim_id')}]"

        return "\n\n".join(
            [
                line("Что решила модель", ["prediction"], "Прогноз недоступен."),
                line("Почему", ["model_rule", "class_concept", "similar_case", "data_quality"], "Объясняющий канал отсутствует."),
                line("Что противоречит", ["forgetting", "subgroup_averaging", "data_deviation", "diagnostic"], "Измеренное противоречие не обнаружено."),
                f"**Применимость.** Получено объяснение уровня {self.explanation_level}. {self.view_model.explanation_level.get('rationale', '')}",
                line("Что делать дальше", ["recommended_action"], f"Действие: {self.action}."),
            ]
        ) + "\n"

    def story(self) -> str:
        """Render the evidence route as data, training, knowledge, decision, action."""

        lines = [f"# История решения ({self.explanation_level})"]
        for stage in self.view_model.visual_spec.get("story", []):
            refs = ", ".join(stage.get("claim_refs", [])) or "нет claims"
            lines.append(f"\n## {stage.get('title')} [{stage.get('status')}] ({refs})")
            facts = stage.get("facts", []) or ["Evidence для этапа отсутствует."]
            lines.extend(f"- {fact}" for fact in facts)
        return "\n".join(lines) + "\n"

    def inspect(self, selector: str) -> ExplanationInspection:
        """Inspect a claim or model rule and return its local provenance."""

        prefix, separator, identifier = selector.partition(":")
        if not separator or prefix not in {"claim", "rule", "evidence", "node"}:
            raise ValueError("selector must be claim:<id>, rule:<id>, or evidence:<node_id>")
        graph = self.view_model.explanation_graph
        nodes = list(graph.get("nodes", []))
        edges = list(graph.get("edges", []))
        claims = list(self.claims)
        target: Mapping[str, Any] | None = None
        anchors: set[str] = set()
        if prefix == "claim":
            normalized = identifier.upper().replace("C", "C-") if identifier.upper().startswith("C") and "-" not in identifier else identifier.upper()
            target = next((claim for claim in claims if str(claim.get("claim_id", "")).upper() == normalized), None)
            if target:
                anchors.add(f"claim:{target.get('claim_id')}")
                anchors.update(str(item) for item in target.get("evidence_refs", []))
        elif prefix == "rule":
            target = next((rule for rule in self.view_model.layers.get("rules", []) if str(rule.get("rule_id")) == identifier), None)
            anchors.add(f"rule:{identifier}")
        else:
            node_id = identifier if prefix == "node" else selector.removeprefix("evidence:")
            target = next((node for node in nodes if str(node.get("node_id")) == node_id), None)
            anchors.add(node_id)
        if target is None:
            raise KeyError(f"unknown inspection target: {selector}")
        related_edges = [edge for edge in edges if edge.get("source") in anchors or edge.get("target") in anchors]
        related_ids = set(anchors)
        for edge in related_edges:
            related_ids.update((str(edge.get("source")), str(edge.get("target"))))
        related_nodes = [node for node in nodes if node.get("node_id") in related_ids]
        related_claims = [
            claim
            for claim in claims
            if f"claim:{claim.get('claim_id')}" in related_ids
            or any(str(ref) in related_ids for ref in claim.get("evidence_refs", []))
        ]
        return ExplanationInspection(selector, target, related_claims, related_nodes, related_edges, self.view_model.visual_spec)

    def audit(self) -> dict[str, Any]:
        """Return full provenance and channel disclosure without presentation text."""

        return {
            "explanation_level": dict(self.view_model.explanation_level),
            "claims": [dict(claim) for claim in self.claims],
            "graph": dict(self.view_model.explanation_graph),
            "diagnostics": list(self.view_model.diagnostics),
            "action": self.action,
            "trace": dict(self.view_model.trace),
            "quality_metrics": dict(self.view_model.quality_metrics),
        }

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
            f"<section><h2>{escape(level.title())}</h2><pre>{escape(self.summary(level))}</pre></section>"
            for level in ("user", "expert", "audit")
            if level in self.view_model.human_explanations
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

    def __init__(self, plan: ExplainPlan | None = None, model_adapter: ModelAdapter | None = None):
        self.plan = plan or ExplainPlan.default()
        self._model_adapter = model_adapter

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
    ) -> "FuzzyXAI":
        """Wrap a callable or predict-proba model with the canonical adapter contract."""

        return cls(plan=explain_plan, model_adapter=resolve_model_adapter(model, adapter))

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
        include_similar_cases: bool = False,
        include_counterfactuals: bool = False,
        include_training_trace: bool = False,
        include_model_knowledge: bool = True,
        additional_evidence: ExplanationEvidence | None = None,
        dataset_version: str = "unversioned",
        run_parameters: Mapping[str, Any] | None = None,
    ) -> ModelExplanationResult:
        """Explain a prediction using only supplied, auditable operator evidence.

        A prediction can always be returned. Gamma, Delta, and rho are computed
        only when their typed evidence sections are present; otherwise the
        result is marked for review instead of receiving synthetic values.
        """

        if self._model_adapter is None:
            raise RuntimeError("FuzzyXAI.explain requires FuzzyXAI.wrap(model, ...)")
        evidence = dict(evidence or {})
        prediction = self._model_adapter.predict(inputs)
        score = prediction.primary_score()
        diagnostics: list[dict[str, Any]] = []
        rows = _as_rows(inputs)
        reference_rows = _as_rows(reference_data) if reference_data is not None else None
        names = list(feature_names or self._model_adapter.feature_names() or _default_feature_names(rows))
        ids = list(object_ids or [f"object_{index}" for index in range(len(rows))])
        internal_evidence = _with_feature_names(self._model_adapter.extract_internal_evidence(inputs), names)

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
            if not rules:
                missing.append("model_rules_or_concepts")
            if reference_rows is not None and reference_labels is not None:
                concepts = build_class_concepts(
                    reference_rows,
                    reference_labels,
                    feature_names=names,
                    object_ids=reference_ids,
                    rules=rules,
                )
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
            missing=list(dict.fromkeys([*missing, *additional.missing])),
        )
        prediction_payload = {**prediction.to_dict(), "score": score}
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
        contribution_method = evidence.get("contribution_method", internal_evidence.get("contribution_method"))
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
            level: compose_human_explanation(
                claims,
                graph,
                action=action,
                level=level,
            ).to_dict()
            for level in ("user", "expert", "audit")
        }
        quality_metrics = evaluate_explanation_quality(
            explanation_evidence,
            graph,
            contributions=dict(evidence.get("contributions", internal_evidence.get("contributions", {}))),
            supplied_metrics=dict(evidence.get("quality_metrics", {})),
        )
        contributions = dict(evidence.get("contributions", internal_evidence.get("contributions", {})))
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
                "adapter_capabilities": self._model_adapter.capabilities(),
                "object_ids": ids,
                "dataset_version": dataset_version,
                "input_sha256": _payload_sha256(rows),
                "generated_at": datetime.now(timezone.utc).isoformat(),
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
        object_id: str,
        include_similar_cases: bool = False,
        include_counterfactuals: bool = False,
        include_training_trace: bool = False,
        **kwargs: Any,
    ) -> ModelExplanationResult:
        """Explain one object while preserving its identifier in every layer."""

        return self.explain(
            _as_rows(input_object),
            object_ids=[object_id],
            include_similar_cases=include_similar_cases,
            include_counterfactuals=include_counterfactuals,
            include_training_trace=include_training_trace,
            **kwargs,
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
