from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import yaml


@dataclass(frozen=True)
class SystemRunFacts:
    dataset_id: str
    split: str
    object_id: str
    source_sha256: str
    preprocessing_trace: dict[str, Any]
    architecture: str
    checkpoint_sha256: str
    model_run_id: str
    probabilities: tuple[float, ...]
    predicted_class: int
    risk_class: int
    source_refs: tuple[str, ...] = ()

    def validate(self) -> None:
        values = np.asarray(self.probabilities, dtype=float)
        if values.ndim != 1 or len(values) < 2 or not np.isfinite(values).all() or np.any(values < 0) or not np.isclose(values.sum(), 1.0, atol=1e-6):
            raise ValueError("registered class probabilities are invalid")
        if self.predicted_class != int(np.argmax(values)) or not 0 <= self.risk_class < len(values):
            raise ValueError("prediction/risk class does not match probability interface")


def load_plan(path: str | Path) -> Any:
    from fuzzyxai.core.explain_plan import ExplainPlan

    raw = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise TypeError("ExplainPlan YAML must be a mapping")
    plan = ExplainPlan.from_dict(raw)
    plan.metadata.update({key: raw[key] for key in ("plan_id", "plan_version", "required_sources", "optional_sources", "not_applicable_rules")})
    return plan


def observation_context(facts: SystemRunFacts, plan: Any, *, critical_fault: dict[str, Any] | None = None, training_run: Any | None = None) -> Any:
    from fuzzyxai import ObservationContext
    from fuzzyxai.core.explanation_object import Trace
    from fuzzyxai.scientific_alignment import AlignmentTransform
    from fuzzyxai.system_semantics import SystemObservation

    facts.validate(); fault = dict(critical_fault or {})
    trace = Trace(id=facts.object_id, version=facts.checkpoint_sha256[:12], timestamp="registered_before_explanation", params={"dataset": facts.dataset_id, "split": facts.split, "model_run_id": facts.model_run_id, "preprocessing": facts.preprocessing_trace}, source="CH6 factual run trace", checksum=facts.source_sha256)
    system = SystemObservation(alignment_transform=AlignmentTransform.from_dict(plan.alignment_policy.transform), risk_membership_policy=plan.membership_policies["system_risk"], risk_class=facts.risk_class, trace=trace, trace_complete=not bool(fault.get("missing_required_trace")), trace_verification_source="CH6 hash/checkpoint verifier", rules_applicable=False, source_refs=(facts.source_sha256, facts.checkpoint_sha256, facts.model_run_id, *facts.source_refs), rupture_present=bool(fault), critical_rupture=bool(fault.get("critical")), rupture_code=str(fault.get("code", "registered_integrity_fault")), rupture_source_refs=tuple(str(value) for value in fault.get("source_refs", (facts.object_id,))))
    return ObservationContext(dataset_version=facts.dataset_id, training_run=training_run, run_parameters={"split": facts.split, "object_id": facts.object_id, "source_sha256": facts.source_sha256, "preprocessing_trace": facts.preprocessing_trace, "architecture": facts.architecture, "checkpoint_sha256": facts.checkpoint_sha256, "model_run_id": facts.model_run_id, "probabilities": list(facts.probabilities), "prediction": facts.predicted_class, "technical_not_clinical": True}, system_observation=system)


def additional_evidence(facts: SystemRunFacts, raw_object: np.ndarray, attribution_maps: list[dict[str, Any]], technical_evidence: dict[str, Any] | None = None) -> Any:
    from fuzzyxai.evidence import DataEvidence, ExplanationEvidence, build_attribution_map

    data = [DataEvidence(object_id=facts.object_id, feature_names=("source_identity", "preprocessing_trace"), raw_values=(facts.source_sha256, facts.preprocessing_trace.get("version")), normalized_values=(None, None), missingness={"source_identity": False, "preprocessing_trace": False}, outlier_scores={"source_identity": None, "preprocessing_trace": None}, anomaly_labels=(), data_quality=1.0, source_trace={"dataset": facts.dataset_id, "split": facts.split, "checkpoint_sha256": facts.checkpoint_sha256, "model_run_id": facts.model_run_id}, warnings=("Technical evidence; not a clinical diagnosis.",), evidence_refs=(facts.source_sha256, facts.checkpoint_sha256))]
    if technical_evidence is not None:
        names = tuple(sorted(technical_evidence)); values = tuple(technical_evidence[name] for name in names)
        data.append(DataEvidence(object_id=f"{facts.object_id}:technical_quality", feature_names=names, raw_values=values, normalized_values=tuple(None for _ in names), missingness={name: value is None for name, value in zip(names, values, strict=True)}, outlier_scores={name: None for name in names}, anomaly_labels=(), data_quality=1.0, source_trace={"method": "registered_technical_quality"}, warnings=("Technical quality evidence is not a medical finding.",), evidence_refs=(facts.source_sha256,)))
    maps = [build_attribution_map(raw_object, item["values"], object_id=facts.object_id, method=str(item["method"]), target=str(item["target"]), baseline=str(item.get("baseline", "not_applicable")), completeness=item.get("completeness", {"status": "not_applicable"}), source_refs=(facts.checkpoint_sha256, facts.object_id, *tuple(item.get("source_refs", ())))) for item in attribution_maps]
    return ExplanationEvidence(data=data, attribution_maps=maps)


def explain_public(model: Any, adapter: Any, numeric_input: Any, raw_object: np.ndarray, facts: SystemRunFacts, *, plan_path: str | Path, attribution_maps: list[dict[str, Any]], technical_evidence: dict[str, Any] | None = None, critical_fault: dict[str, Any] | None = None, training_run: Any | None = None, feature_names: list[str] | None = None) -> Any:
    from fuzzyxai import FuzzyXAI

    plan = load_plan(plan_path); context = observation_context(facts, plan, critical_fault=critical_fault, training_run=training_run); evidence = additional_evidence(facts, raw_object, attribution_maps, technical_evidence)
    result = FuzzyXAI.wrap(model, adapter=adapter, explain_plan=plan, observation_context=context).explain_one(numeric_input, object_id=facts.object_id, raw_object=raw_object, additional_evidence=evidence, dataset_version=facts.dataset_id, feature_names=feature_names)
    if result.system is None:
        reasons = [
            str(item.get("reason", item))
            for item in result.view_model.diagnostics
            if item.get("code") == "D_system_route_incomplete"
        ]
        raise RuntimeError(
            "public explain_one did not create SystemEvidence"
            + (f": {'; '.join(reasons)}" if reasons else "")
        )
    return result
