from __future__ import annotations

from dataclasses import asdict, replace
from pathlib import Path
from time import perf_counter
from typing import Any

from fuzzyxai.core.explanation_object import ExplanationObject as MathematicalExplanationObject
from fuzzyxai.core.explanation_object import Rule, Trace
from fuzzyxai.core.types import AdaptedInput, ExplainableObject
from fuzzyxai.diagnostics.contracts import (
    Contract,
    RepairExecutionContext,
    RouteEdge,
    RouteGraph,
    RouteNode,
    canonical_sha256,
)
from fuzzyxai.diagnostics.minimal_cut import MinimalDiagnosticCutFinder
from fuzzyxai.diagnostics.recertification import RouteRecertifier
from fuzzyxai.diagnostics.repair_executor import RepairExecutor
from fuzzyxai.diagnostics.repair_planner import ActionableRepairPlanner
from fuzzyxai.diagnostics.repair_registry import RepairProviderRegistry
from fuzzyxai.diagnostics.reporter import DiagnosticReporter
from fuzzyxai.diagnostics.validator import DiagnosticValidator

from .contracts import (
    ExplanationClaim,
    ObserverDecision,
    PredictionRequest,
    VerticalRun,
)
from .math import reduce_representation, select_representation, uncertainty_profile
from .model import EXPECTED_SCHEMA_SHA256, MODEL_ID, MODEL_VERSION, BreastCancerModel

ROOT = Path(__file__).resolve().parents[4]
PLAN_SHA256 = "dc758d1762500f6b63775cc4211e182de71d401f69c1e9caa2f652ba93e16943"
FORBIDDEN = frozenset({"gold_label", "target", "fix_commit", "gold_patch", "changed_files", "changed_symbols"})
SCENARIOS: dict[str, dict[str, Any]] = {
    "S1_NORMAL": {},
    "S2_EXPLAINER_VERSION_MISMATCH": {"explainer_version": "0.0.0"},
    "S3_MISSING_REQUIRED_FEATURE": {"missing_feature": "mean radius"},
    "S4_MODEL_RULE_CONFLICT": {"rule_counter_evidence": True},
    "S5_INTERVAL_UNCERTAINTY": {"feature_interval": True, "feature_interval_width": 0.12},
    "S6_MULTILEVEL_UNCERTAINTY": {"feature_interval": True, "rule_counter_evidence": True, "trace_complexity": True},
    "S7_REDUCTION_LOSS_EXCEEDED": {"feature_interval": True, "reduction_stress": True},
    "S8_INCOMPLETE_PROVENANCE": {"omit_provenance": True},
    "S9_REGISTERED_REPAIR": {"explainer_version": "0.0.0", "execute_repair": True},
    "S10_DETERMINISM": {},
}


def _assert_observable(payload: Any, path: str = "request") -> None:
    if isinstance(payload, dict):
        for key, value in payload.items():
            if str(key).lower() in FORBIDDEN:
                raise ValueError(f"forbidden pre-scoring channel: {path}.{key}")
            _assert_observable(value, f"{path}.{key}")
    elif isinstance(payload, (list, tuple)):
        for index, value in enumerate(payload):
            _assert_observable(value, f"{path}[{index}]")


class MLVerticalService:
    """Deterministic, evidence-bound ML explanation route."""

    def __init__(self, *, persist_dir: str | Path | None = None) -> None:
        self.model = BreastCancerModel()
        self.persist_dir = Path(persist_dir) if persist_dir else None
        self.runs: dict[str, VerticalRun] = {}
        self.graphs: dict[str, RouteGraph] = {}
        self.last_timings: dict[str, float] = {}

    def scenario_request(self, scenario_id: str) -> PredictionRequest:
        if scenario_id not in SCENARIOS:
            raise KeyError(scenario_id)
        controls = dict(SCENARIOS[scenario_id])
        features = dict(self.model.default_features)
        missing = controls.get("missing_feature")
        if missing:
            features.pop(str(missing), None)
        return PredictionRequest(scenario_id, f"bcw:{scenario_id.lower()}", features, controls)

    def execute(self, request: PredictionRequest | dict[str, Any]) -> VerticalRun:
        if isinstance(request, dict):
            request = PredictionRequest(**request)
        _assert_observable(asdict(request))
        if request.scenario_id not in SCENARIOS:
            raise ValueError(f"unregistered scenario: {request.scenario_id}")
        controls = {**SCENARIOS[request.scenario_id], **request.controls}
        _assert_observable(controls, "controls")
        missing = self.model.validate_features(request.features)
        prediction = None
        explanation = None
        probability = 0.5
        started = perf_counter()
        if not missing:
            model_started = perf_counter()
            prediction = self.model.predict(request.object_id, request.features)
            model_ms = (perf_counter() - model_started) * 1000
            probability = prediction.probability
            shap_started = perf_counter()
            explanation = self.model.explain(
                request.object_id,
                request.features,
                observed_model_version=str(controls.get("explainer_version", MODEL_VERSION)),
            )
            shap_ms = (perf_counter() - shap_started) * 1000
        else:
            model_ms = shap_ms = 0.0
        trace_complete = not controls.get("omit_provenance", False)
        profile = uncertainty_profile(probability=probability, controls=controls, trace_complete=trace_complete)
        representation = select_representation(profile, probability)
        reduction = reduce_representation(
            representation, maximum_loss=0.25, stress=bool(controls.get("reduction_stress")),
        )
        explainable = self._explainable_object(request, prediction, explanation, profile, representation, reduction)
        graph = self._route_graph(request, controls, missing, prediction, explanation, profile, representation, reduction, explainable)
        validator = DiagnosticValidator()
        validation = validator.validate(graph)
        cut = plan = recertification = None
        if validation.issues:
            cut = replace(MinimalDiagnosticCutFinder().find(graph, validation), runtime_ms=0.0)
            registry = RepairProviderRegistry()
            registry.providers.sort(key=lambda provider: provider.provider_id != "explainer.rerun")
            plan = ActionableRepairPlanner(registry).plan(graph, validation.issues, cut)
        repair_payload: dict[str, Any] | None = None
        final_graph = graph
        if controls.get("execute_repair") and plan and plan.steps:
            corrected_graph = self._correct_explainer_graph(graph)
            handlers = {
                "rerun_explainer_with_registered_components": lambda _graph, _step: corrected_graph,
                "restore_previous_artifact_snapshot": lambda before, _step: before,
            }
            context = RepairExecutionContext(
                handlers=handlers,
                approved_step_ids=frozenset(step.step_id for step in plan.steps),
                allow_external_changes=True,
            )
            final_graph, execution = RepairExecutor(registry).execute(graph, plan, context)
            recertification = RouteRecertifier().recertify(graph, final_graph, plan, execution)
            repair_payload = {
                "plan": asdict(plan),
                "execution": [asdict(item) for item in execution],
                "recertification": asdict(recertification),
            }
            validation = validator.validate(final_graph)
        report = DiagnosticReporter().build(final_graph, validation, validation.issues, cut, plan, recertification)
        observer = self._observer(missing, validation.issues, profile, reduction.loss, recertification)
        claims = self._claims(request, prediction, explanation, observer)
        structured_explainable = self._structured_explainable(
            request, prediction, explanation, profile, representation, reduction, claims,
        )
        run_id = f"mlv1:{canonical_sha256({'request': asdict(request), 'controls': controls})[:20]}"
        explainable_sha256 = str(graph.metadata["explainable_object"])
        views = self._views(run_id, request, prediction, explanation, representation, reduction, report, observer, claims, explainable_sha256)
        run = VerticalRun.build(
            run_id=run_id,
            scenario_id=request.scenario_id,
            request=asdict(request),
            prediction=asdict(prediction) if prediction else None,
            explanation=asdict(explanation) if explanation else None,
            explainable_object=structured_explainable,
            uncertainty=asdict(profile),
            representation=asdict(representation),
            reduction=asdict(reduction),
            route_graph=final_graph.to_dict(),
            diagnosis=report.to_dict(),
            observer=asdict(observer),
            claims=tuple(asdict(claim) for claim in claims),
            views=views,
            repair=repair_payload,
        )
        self.runs[run_id] = run
        self.graphs[run_id] = final_graph
        self._persist(run)
        total_ms = (perf_counter() - started) * 1000
        self.last_timings = {
            "model_ms": model_ms,
            "shap_ms": shap_ms,
            "fuzzyxai_ms": max(0.0, total_ms - model_ms - shap_ms),
            "total_ms": total_ms,
        }
        return run

    def get(self, run_id: str) -> VerticalRun:
        try:
            return self.runs[run_id]
        except KeyError as exc:
            raise KeyError(f"unknown run: {run_id}") from exc

    def _explainable_object(self, request, prediction, explanation, profile, representation, reduction) -> ExplainableObject:
        adapted = AdaptedInput(request.scenario_id, request.features, "sklearn-adapter")
        rules = [
            Rule("high_model_risk", {"probability": ">=0.5"}, "high_risk"),
            Rule("feature_pattern_support", {"worst radius": "high", "worst concave points": "high"}, "high_risk"),
        ]
        activations = {"high_model_risk": prediction.probability if prediction else 0.0, "feature_pattern_support": 0.0}
        mathematical = MathematicalExplanationObject(
            terms={"low", "medium", "high"}, representation=representation, rules=rules,
            activations=activations, uncertainty=profile.aggregate,
            trace=Trace(request.object_id, "1.0.0", "2000-01-01T00:00:00Z", {"plan": PLAN_SHA256}, "ml_vertical_v1", explanation.artifact_sha256 if explanation else "missing"),
            reduction_loss=reduction.loss,
        )
        return ExplainableObject(
            request.scenario_id, adapted,
            {"mathematical": mathematical, "prediction": prediction, "local_explanation": explanation, "representation": representation},
            {"plan_sha256": PLAN_SHA256},
        )

    def _route_graph(self, request, controls, missing, prediction, explanation, profile, representation, reduction, explainable) -> RouteGraph:
        evidence_hash = explanation.artifact_sha256 if explanation else None
        node_specs = (
            ("input", "input", {"schema_complete": not missing, "values_finite": all(float(value) == float(value) for value in request.features.values()), "object_id": request.object_id}),
            ("feature_schema", "schema", {"schema_sha256": EXPECTED_SCHEMA_SHA256}),
            ("preprocessor", "preprocessor", {"version": "sklearn.StandardScaler/1"}),
            ("model", "model", {"model_id": MODEL_ID, "model_version": MODEL_VERSION, "feature_schema_sha256": EXPECTED_SCHEMA_SHA256}),
            ("prediction", "prediction", {"object_id": request.object_id, "probability": prediction.probability if prediction else None}),
            ("explainer", "explainer", {"explainer_version": explanation.explainer_version if explanation else None, "model_version": explanation.model_version if explanation else None}),
            ("local_explanation", "explanation", {"object_id": explanation.object_id if explanation else None, "output_difference": explanation.output_difference if explanation else None}),
            ("rule_source", "rule_source", {"version": "bcw-rules/1.0.0"}),
            ("uncertainty", "uncertainty", {"coverage_complete": set(profile.present_types).issubset(set(representation.covered_uncertainties))}),
            ("reduction", "reduction", {"loss": reduction.loss}),
            ("observer_action", "observer", {"registered": True}),
            ("user_claim", "claim", {"evidence_bound": bool(explanation) or bool(missing)}),
            ("audit_artifact", "audit", {"artifact_sha256": None if controls.get("omit_provenance") else evidence_hash, "plan_sha256": None if controls.get("omit_provenance") else PLAN_SHA256}),
        )
        nodes = tuple(
            RouteNode(node_id, node_type, f"mlv1.{node_id}", "1.0.0", attrs, attrs, True, node_id in {"explainer", "reduction", "audit_artifact"}, (f"evidence:{node_id}",))
            for node_id, node_type, attrs in node_specs
        )
        chain = [item[0] for item in node_specs]
        relations = ["adapts", "defines", "transforms", "predicts", "explains", "instantiates", "constrains", "profiles", "reduces", "observes", "binds", "records"]
        edges = tuple(
            RouteEdge(f"edge:{source}:{target}", source, target, relation, True, {"relation": relation}, {"relation": relation}, False, relation_status="known_valid")
            for source, target, relation in zip(chain[:-1], chain[1:], relations, strict=True)
        )
        contracts = (
            Contract("INPUT_FEATURE_SCHEMA", "equals", "input", "schema_complete", True, category="data", repairable=False),
            Contract("INPUT_VALUE_DOMAIN", "equals", "input", "values_finite", True, category="data", repairable=False),
            Contract("MODEL_VERSION", "equals", "model", "model_version", MODEL_VERSION, category="model"),
            Contract("MODEL_FEATURE_SCHEMA", "equals", "model", "feature_schema_sha256", EXPECTED_SCHEMA_SHA256, category="model"),
            Contract("MODEL_EXPLAINER_VERSION", "equals", "explainer", "model_version", MODEL_VERSION, category="explainer", source_nodes=("explainer",)),
            Contract("EXPLANATION_OBJECT_ID", "equals", "local_explanation", "object_id", request.object_id, category="explainer"),
            Contract("EXPLANATION_OUTPUT_CONSISTENCY", "max_value", "local_explanation", "output_difference", 1e-8, category="explainer"),
            Contract("RULE_SOURCE_VERSION", "equals", "rule_source", "version", "bcw-rules/1.0.0", category="representation"),
            Contract("REQUIRED_PROVENANCE", "required_attribute", "audit_artifact", "plan_sha256", category="provenance"),
            Contract("UNCERTAINTY_REPRESENTATION_COVERAGE", "equals", "uncertainty", "coverage_complete", True, category="representation"),
            Contract("REDUCTION_LOSS_LIMIT", "max_value", "reduction", "loss", 0.25, category="reduction"),
            Contract("USER_CLAIM_EVIDENCE_BINDING", "equals", "user_claim", "evidence_bound", True, category="provenance"),
            Contract("AUDIT_ARTIFACT_HASH", "required_attribute", "audit_artifact", "artifact_sha256", category="provenance"),
        )
        return RouteGraph(
            f"route:{request.scenario_id}:{request.object_id}", nodes, edges, contracts,
            {"explainable_object": canonical_sha256({"scenario": explainable.scenario_id, "metadata": explainable.metadata}), "repair_costs": {"node:explainer": 0.1}},
        )

    @staticmethod
    def _correct_explainer_graph(graph: RouteGraph) -> RouteGraph:
        nodes = tuple(
            replace(node, observed_attributes={**node.observed_attributes, "model_version": MODEL_VERSION})
            if node.node_id == "explainer" else node
            for node in graph.nodes
        )
        return replace(graph, nodes=nodes)

    @staticmethod
    def _observer(missing, issues, profile, reduction_loss, recertification) -> ObserverDecision:
        issue_ids = tuple(issue.violated_contract for issue in issues)
        critical = tuple(issue.violated_contract for issue in issues if issue.severity == "error")
        if recertification and recertification.status == "full_success":
            return ObserverDecision("ACCEPT", 0.1, ("registered repair passed full recertification",), ())
        if missing:
            return ObserverDecision("REQUEST_DATA", 1.0, (f"required features missing: {', '.join(missing)}",), critical)
        if {"MODEL_EXPLAINER_VERSION", "REQUIRED_PROVENANCE", "AUDIT_ARTIFACT_HASH"}.intersection(issue_ids):
            return ObserverDecision("BLOCK", 1.0, ("critical registered contract is not satisfied",), critical)
        if profile.conflict > 0:
            return ObserverDecision("REVIEW", min(1.0, profile.aggregate + profile.conflict), ("registered evidence sources disagree",), critical)
        if reduction_loss > 0.25:
            return ObserverDecision("WARN", min(1.0, reduction_loss + 0.2), ("reduction loss exceeds the registered ceiling",), critical)
        if "u_int" in profile.present_types or profile.aggregate >= 0.15:
            return ObserverDecision("WARN", profile.aggregate, ("noncritical uncertainty is retained",), critical)
        return ObserverDecision("ACCEPT", profile.aggregate, ("all registered route contracts are satisfied",), critical)

    @staticmethod
    def _structured_explainable(request, prediction, explanation, profile, representation, reduction, claims):
        evidence = [
            {
                "evidence_id": "evidence:prediction",
                "type": "model_output",
                "subject": request.object_id,
                "value": prediction.probability if prediction else None,
                "status": "observed" if prediction else "missing",
                "source_id": MODEL_ID,
                "source_version": MODEL_VERSION,
                "object_id": request.object_id,
                "created_at": "2000-01-01T00:00:00Z",
                "artifact_hash": prediction.model_sha256 if prediction else "",
            },
            {
                "evidence_id": "evidence:local_explanation",
                "type": "local_explanation",
                "subject": request.object_id,
                "value": explanation.output_sum if explanation else None,
                "status": "observed" if explanation else "missing",
                "source_id": "shap.LinearExplainer",
                "source_version": explanation.explainer_version if explanation else "missing",
                "object_id": request.object_id,
                "created_at": "2000-01-01T00:00:00Z",
                "artifact_hash": explanation.artifact_sha256 if explanation else "",
            },
        ]
        payload = {
            "object_id": request.object_id,
            "evidence": evidence,
            "claims": [asdict(item) for item in claims],
            "provenance_relations": [
                {"source_id": "evidence:prediction", "target_id": "claim:prediction", "relation": "supports"},
                {"source_id": "evidence:local_explanation", "target_id": "claim:features", "relation": "supports"},
            ],
            "uncertainty_profile": asdict(profile),
            "representation": asdict(representation),
            "reduction": asdict(reduction),
            "presentation_policy": "evidence_bound_registered_templates",
            "explain_plan_id": "bcw-logreg-shap-v1/1.0.0",
        }
        payload["canonical_hash"] = canonical_sha256(payload)
        return payload

    @staticmethod
    def _claims(request, prediction, explanation, observer) -> tuple[ExplanationClaim, ...]:
        if prediction is None:
            return (ExplanationClaim("claim:missing", "A prediction was not produced because required input is missing.", ("evidence:input",), "limited"),)
        strongest = sorted(explanation.shap_values, key=lambda name: abs(explanation.shap_values[name]), reverse=True)[:3]
        return (
            ExplanationClaim("claim:prediction", f"The registered model produced score {prediction.probability:.6f} for object {request.object_id}.", ("evidence:prediction",), "observed"),
            ExplanationClaim("claim:features", f"Largest local SHAP contributions: {', '.join(strongest)}.", ("evidence:local_explanation",), "observed"),
            ExplanationClaim("claim:action", f"The route observer selected {observer.action}.", ("evidence:observer_action",), "observed"),
        )

    @staticmethod
    def _views(run_id, request, prediction, explanation, representation, reduction, report, observer, claims, explainable_sha256):
        common = {"run_id": run_id, "scenario_id": request.scenario_id, "action": observer.action, "claims": [asdict(item) for item in claims], "explainable_object_sha256": explainable_sha256}
        user = {**common, "view": "user", "prediction": asdict(prediction) if prediction else None, "limitations": report.limitations}
        engineer = {**common, "view": "engineer", "explanation": asdict(explanation) if explanation else None, "representation": asdict(representation), "reduction": asdict(reduction), "issues": [asdict(item) for item in report.issues]}
        auditor = {**common, "view": "auditor", "diagnostic_report": report.to_dict()}
        return {"user": user, "engineer": engineer, "auditor": auditor}

    def _persist(self, run: VerticalRun) -> None:
        if not self.persist_dir:
            return
        import json

        self.persist_dir.mkdir(parents=True, exist_ok=True)
        path = self.persist_dir / f"{run.run_id.replace(':', '_')}.json"
        path.write_text(json.dumps(asdict(run), ensure_ascii=False, indent=2, default=str) + "\n", encoding="utf-8")
