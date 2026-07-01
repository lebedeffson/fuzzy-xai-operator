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
    return ValidationResult(not errors, errors, schema)


def validate_file(path: str | Path, schema: str) -> ValidationResult:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        return ValidationResult(False, ["JSON root must be object"], schema)
    return validate_payload(data, schema)
