from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class ValidationResult:
    valid: bool
    errors: list[str] = field(default_factory=list)
    schema: str = ""


REQUIRED: dict[str, tuple[str, ...]] = {
    "classification": (
        "scenario_id",
        "source_type",
        "model_name",
        "dataset_name",
        "predicted_class",
        "class_probability",
        "feature_values",
        "feature_importance",
        "quality_metrics",
    ),
    "regression": (
        "scenario_id",
        "source_type",
        "model_name",
        "dataset_name",
        "prediction",
        "feature_values",
        "feature_importance",
        "quality_metrics",
    ),
    "signal": (
        "scenario_id",
        "source_type",
        "model_name",
        "signal_quality",
        "quality_metrics",
        "feature_values",
    ),
    "image": (
        "scenario_id",
        "source_type",
        "model_name",
        "dataset_name",
        "feature_values",
        "feature_importance",
        "quality_metrics",
    ),
    "route": ("scenario_id", "nodes", "computed_result", "final_action"),
    "proof_trace": ("package_type", "scenario_id", "route", "computed_result", "final_action"),
    "operator_trace": ("scenario_id", "nodes", "edges", "computed_result"),
    "explanation_view_model": (
        "schema_version",
        "model",
        "route",
        "risk",
        "diagnostics",
        "trace",
        "layers",
        "explanation_graph",
        "human_explanations",
    ),
    "explanation_visual_spec": (
        "schema_version",
        "overview",
        "story",
        "data_profile",
        "training_timeline",
        "knowledge_atlas",
        "decision_evidence",
        "similar_cases",
        "counterfactuals",
        "rule_ablations",
        "provenance_nodes",
        "provenance_edges",
        "audit",
    ),
    "human_explanation": (
        "audience",
        "language",
        "decision",
        "main_reasons",
        "concerns",
        "reliability",
        "recommended_action",
        "what_would_change_result",
        "details",
    ),
}


def list_schemas() -> list[str]:
    return sorted(REQUIRED)


def validate_payload(payload: dict[str, Any], schema: str) -> ValidationResult:
    if schema not in REQUIRED:
        return ValidationResult(False, [f"unknown schema: {schema}"], schema)
    errors = [f"missing field: {field}" for field in REQUIRED[schema] if field not in payload]
    if schema == "classification" and "class_probability" in payload:
        try:
            probability = float(payload["class_probability"])
            if not 0.0 <= probability <= 1.0:
                errors.append("class_probability must be in [0,1]")
        except Exception:
            errors.append("class_probability must be numeric")
    for object_field in ("feature_values", "feature_importance", "quality_metrics"):
        if object_field in payload and not isinstance(payload[object_field], dict):
            errors.append(f"{object_field} must be object")
    if schema in {"route", "operator_trace"} and "nodes" in payload and not isinstance(payload["nodes"], list):
        errors.append("nodes must be array")
    if schema == "explanation_view_model":
        if payload.get("schema_version") != "2.0":
            errors.append("schema_version must be 2.0")
        for object_field in ("model", "risk", "trace", "layers", "explanation_graph", "human_explanations"):
            if object_field in payload and not isinstance(payload[object_field], dict):
                errors.append(f"{object_field} must be object")
    if schema == "explanation_visual_spec":
        if payload.get("schema_version") != "1.1":
            errors.append("schema_version must be 1.1")
        for object_field in ("overview", "knowledge_atlas", "decision_evidence", "audit"):
            if object_field in payload and not isinstance(payload[object_field], dict):
                errors.append(f"{object_field} must be object")
        for array_field in ("story", "data_profile", "training_timeline", "similar_cases", "counterfactuals", "rule_ablations", "provenance_nodes", "provenance_edges"):
            if array_field in payload and not isinstance(payload[array_field], (list, tuple)):
                errors.append(f"{array_field} must be array")
    if schema == "human_explanation":
        if payload.get("audience") not in {"domain_user", "ml_engineer", "researcher", "auditor"}:
            errors.append("audience must be domain_user, ml_engineer, researcher, or auditor")
        for object_field in ("decision", "reliability", "recommended_action", "details"):
            if object_field in payload and not isinstance(payload[object_field], dict):
                errors.append(f"{object_field} must be object")
        for array_field in ("main_reasons", "concerns", "what_would_change_result"):
            if array_field in payload and not isinstance(payload[array_field], (list, tuple)):
                errors.append(f"{array_field} must be array")
    return ValidationResult(not errors, errors, schema)


def validate_file(path: str | Path, schema: str) -> ValidationResult:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        return ValidationResult(False, ["JSON root must be object"], schema)
    return validate_payload(data, schema)
