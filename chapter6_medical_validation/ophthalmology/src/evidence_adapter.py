from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import yaml


@dataclass(frozen=True)
class MedicalRunFacts:
    dataset_id: str
    split: str
    sample_id: str
    raw_image_sha256: str
    preprocessing_trace: dict[str, Any]
    architecture: str
    checkpoint_sha256: str
    model_run_id: str
    full_probabilities: tuple[float, float, float, float, float]
    predicted_grade: int
    calibration_status: str
    technical_quality: dict[str, Any]

    @property
    def referable_dr_probability(self) -> float:
        return float(sum(self.full_probabilities[2:]))

    def validate(self) -> None:
        probabilities = np.asarray(self.full_probabilities, dtype=float)
        if probabilities.shape != (5,) or not np.isfinite(probabilities).all():
            raise ValueError("medical run requires five finite probabilities")
        if np.any(probabilities < 0) or not np.isclose(probabilities.sum(), 1.0, atol=1e-6):
            raise ValueError("medical probabilities must be non-negative and sum to one")
        if self.predicted_grade != int(np.argmax(probabilities)):
            raise ValueError("predicted grade must equal argmax of registered probabilities")


def load_eye_explain_plan(path: str | Path) -> Any:
    from fuzzyxai.core.explain_plan import ExplainPlan

    raw = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise TypeError("eye ExplainPlan must be a mapping")
    plan = ExplainPlan.from_dict(raw)
    plan.metadata.update({
        "plan_id": raw["plan_id"],
        "plan_version": raw["plan_version"],
        "dataset_scope": list(raw["dataset_scope"]),
        "required_sources": list(raw["required_sources"]),
        "optional_sources": list(raw["optional_sources"]),
        "not_applicable_rules": list(raw["not_applicable_rules"]),
    })
    return plan


def build_observation_context(facts: MedicalRunFacts, plan: Any, *, critical_fault: dict[str, Any] | None = None) -> Any:
    from fuzzyxai import ObservationContext
    from fuzzyxai.core.explanation_object import Trace
    from fuzzyxai.scientific_alignment import AlignmentTransform
    from fuzzyxai.system_semantics import SystemObservation

    facts.validate()
    transform = AlignmentTransform.from_dict(plan.alignment_policy.transform)
    membership = plan.membership_policies["system_risk"]
    trace_payload = facts.preprocessing_trace
    trace = Trace(
        id=facts.sample_id,
        version=facts.checkpoint_sha256[:12],
        timestamp=str(trace_payload.get("captured_at", "registered_before_explanation")),
        params={
            "dataset": facts.dataset_id,
            "split": facts.split,
            "model_run_id": facts.model_run_id,
            "preprocessing_config_sha256": trace_payload.get("config_sha256"),
        },
        source="CH6 ophthalmology factual run trace",
        checksum=facts.raw_image_sha256,
    )
    fault = dict(critical_fault or {})
    observation = SystemObservation(
        alignment_transform=transform,
        risk_membership_policy=membership,
        risk_class=4,
        trace=trace,
        trace_complete=not bool(fault.get("missing_required_trace")),
        trace_verification_source="CH6 loader + checkpoint/preprocessing hash verifier",
        rules_applicable=False,
        source_refs=(facts.raw_image_sha256, facts.checkpoint_sha256, facts.model_run_id),
        rupture_present=bool(fault),
        critical_rupture=bool(fault.get("critical")),
        rupture_code=str(fault.get("code", "ch6_eye_registered_integrity_fault")),
        rupture_source_refs=tuple(str(value) for value in fault.get("source_refs", (facts.sample_id,))),
    )
    return ObservationContext(
        dataset_version=facts.dataset_id,
        run_parameters={
            "split": facts.split,
            "sample_id": facts.sample_id,
            "raw_image_sha256": facts.raw_image_sha256,
            "preprocessing_trace": facts.preprocessing_trace,
            "architecture": facts.architecture,
            "checkpoint_sha256": facts.checkpoint_sha256,
            "model_run_id": facts.model_run_id,
            "full_probabilities": list(facts.full_probabilities),
            "predicted_grade": facts.predicted_grade,
            "referable_dr_probability": facts.referable_dr_probability,
            "calibration_status": facts.calibration_status,
            "technical_quality": facts.technical_quality,
            "medical_claim_boundary": "model output, not patient diagnosis",
        },
        system_observation=observation,
    )


def build_additional_evidence(
    facts: MedicalRunFacts,
    raw_rgb: np.ndarray,
    *,
    grad_cam_map: np.ndarray,
    grad_cam_metadata: dict[str, Any],
    lesion_masks: dict[str, np.ndarray] | None = None,
) -> Any:
    from fuzzyxai.evidence import DataEvidence, ExplanationEvidence, build_attribution_map

    facts.validate()
    quality_keys = (
        "blur_laplacian_variance",
        "underexposure_fraction",
        "overexposure_fraction",
        "field_of_view_coverage",
    )
    quality_values = [facts.technical_quality.get(key) for key in quality_keys]
    source_trace = {
        "dataset": facts.dataset_id,
        "split": facts.split,
        "raw_image_sha256": facts.raw_image_sha256,
        "preprocessing": facts.preprocessing_trace,
        "model_run_id": facts.model_run_id,
        "checkpoint_sha256": facts.checkpoint_sha256,
    }
    data_records = [
        DataEvidence(
            object_id=facts.sample_id,
            feature_names=("raw_image_identity", "preprocessing_trace"),
            raw_values=(facts.raw_image_sha256, facts.preprocessing_trace.get("output_sha256")),
            normalized_values=(None, None),
            missingness={"raw_image_identity": False, "preprocessing_trace": False},
            outlier_scores={"raw_image_identity": None, "preprocessing_trace": None},
            anomaly_labels=(),
            data_quality=1.0,
            source_trace=source_trace,
            warnings=("Evidence identifies an image/model route; it is not a clinical-quality statement.",),
            evidence_refs=(facts.raw_image_sha256, facts.checkpoint_sha256),
        ),
        DataEvidence(
            object_id=f"{facts.sample_id}:technical_quality",
            feature_names=quality_keys,
            raw_values=quality_values,
            normalized_values=tuple(float(value) if value is not None else None for value in quality_values),
            missingness={key: facts.technical_quality.get(key) is None for key in quality_keys},
            outlier_scores={key: None for key in quality_keys},
            anomaly_labels=(),
            data_quality=float(facts.technical_quality.get("quality_score", 1.0)),
            source_trace={"method": "technical_image_quality_evidence", "parameters": facts.technical_quality.get("parameters", {})},
            warnings=("Technical metrics are not a clinical assessment of image quality.",),
            evidence_refs=(facts.raw_image_sha256,),
        ),
    ]
    grad_cam = build_attribution_map(
        raw_rgb,
        grad_cam_map,
        object_id=facts.sample_id,
        method="grad_cam",
        target=str(grad_cam_metadata["target_class"]),
        baseline="not_applicable",
        completeness={"status": "not_applicable", "reason": "Grad-CAM has no IG completeness identity"},
        source_refs=(facts.checkpoint_sha256, str(grad_cam_metadata["target_layer"]), facts.sample_id),
    )
    missing: list[str] = []
    if lesion_masks is None:
        missing.append("optional_idrid_lesion_masks")
    return ExplanationEvidence(data=data_records, attribution_maps=[grad_cam], missing=missing)


def run_public_explanation(
    model: Any,
    adapter: Any,
    numeric_input: Any,
    raw_rgb: np.ndarray,
    facts: MedicalRunFacts,
    *,
    plan_path: str | Path,
    grad_cam_map: np.ndarray,
    grad_cam_metadata: dict[str, Any],
    lesion_masks: dict[str, np.ndarray] | None = None,
    critical_fault: dict[str, Any] | None = None,
) -> Any:
    from fuzzyxai import FuzzyXAI

    plan = load_eye_explain_plan(plan_path)
    context = build_observation_context(facts, plan, critical_fault=critical_fault)
    additional = build_additional_evidence(
        facts,
        raw_rgb,
        grad_cam_map=grad_cam_map,
        grad_cam_metadata=grad_cam_metadata,
        lesion_masks=lesion_masks,
    )
    result = FuzzyXAI.wrap(
        model,
        adapter=adapter,
        explain_plan=plan,
        observation_context=context,
    ).explain_one(
        numeric_input,
        object_id=facts.sample_id,
        raw_object=raw_rgb,
        additional_evidence=additional,
        region_masks=lesion_masks,
        dataset_version=facts.dataset_id,
    )
    if result.system is None:
        raise RuntimeError("public explain_one returned no SystemEvidence")
    return result


def export_public_result(result: Any, output_dir: str | Path) -> None:
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    result.export_json(output / "result.json", detail="audit")
    (output / "audit.json").write_text(json.dumps(result.audit(), ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    (output / "full_report_reader_ru.txt").write_text(result.full_report(level="reader"), encoding="utf-8")
    (output / "full_report_audit_ru.txt").write_text(result.full_report(level="audit"), encoding="utf-8")
    (output / "provenance.json").write_text(json.dumps(result.view_model.explanation_graph, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    result.visualize(view="provenance", selector="action", output=str(output / "provenance_action.png"))
