from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from scripts.merge_model_validation_reports import merge_reports, validate_runtime_report


def _write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _base_evidence(path: Path) -> Path:
    base = path / "base"
    _write_json(base / "summary.json", {"sklearn_version": "1.7.0"})
    report = {
        "config_id": "logistic_binary",
        "task": "binary_classification",
        "model_class": "sklearn.linear_model.LogisticRegression",
        "adapter_id": "sklearn_linear_v2",
        "model_family": "linear",
        "prediction_parity": True,
        "conformance": {"status": "pass"},
        "capabilities": {"capabilities": {}},
        "graph_errors": [],
        "human_explanation": {
            "decision": {"explanation": "decision"},
            "reliability": {"explanation": "reliable"},
            "recommended_action": {"explanation": "review"},
        },
        "quality": {"status": "pass"},
        "missing_channels": [],
        "status": "pass",
    }
    _write_json(base / "conformance_reports/logistic_binary.json", report)
    return base


def _runtime_report(library: str = "xgboost", *, python_version: str = "3.11.9", status: str = "pass") -> dict[str, object]:
    payload: dict[str, object] = {
        "schema_version": "1.0",
        "report_id": f"{library}_py{python_version.replace('.', '')[:3]}",
        "library": library,
        "library_version": "2.1.0",
        "python_version": python_version,
        "operating_system": "Linux",
        "environment": f"optional-runtime-{library}",
        "model_family": library,
        "model_class": "Classifier",
        "task_type": "binary_classification",
        "adapter": "Adapter",
        "adapter_id": f"{library}_v2",
        "sample_size": 64,
        "prediction_parity": 1.0 if status == "pass" else 0.0,
        "conformance": 1.0,
        "graph_validation": 1.0,
        "quality_gate": "pass",
        "quality": {"completeness": 1.0, "measured_top_reason_stability": 1.0, "limitations": []},
        "api_checks": {"explain_one": True, "explain_batch": True, "explain_global": True, "why_not": True, "compare_models": True},
        "human_explanation_checks": {
            "decision_present": True,
            "reason_count_valid": True,
            "concern_count_valid": True,
            "reliability_present": True,
            "recommended_action_present": True,
            "comparisons_present": True,
            "all_fragments_grounded": True,
            "technical_terms_hidden": True,
        },
        "native_channels": ["prediction"],
        "derived_channels": [],
        "surrogate_channels": [],
        "missing_channels": [],
        "duration_seconds": 0.1,
        "warnings": [],
        "status": status,
    }
    canonical = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"), default=str).encode("utf-8")
    payload["artifact_sha256"] = hashlib.sha256(canonical).hexdigest()
    return payload


def test_missing_runtime_report_never_becomes_pass(tmp_path: Path) -> None:
    base = _base_evidence(tmp_path)
    summary = merge_reports(base, [], tmp_path / "merged", tmp_path / "quality")
    assert summary["status_counts"]["implemented_not_executed"] == 6
    matrix = json.loads((tmp_path / "merged/model_support_matrix.json").read_text(encoding="utf-8"))
    optional = [row for row in matrix["configurations"] if row["library"] == "xgboost"]
    assert optional[0]["status"] == "implemented_not_executed"


def test_valid_runtime_report_promotes_only_measured_library(tmp_path: Path) -> None:
    base = _base_evidence(tmp_path)
    report_path = tmp_path / "adapter_report_xgboost_py311.json"
    _write_json(report_path, _runtime_report())
    summary = merge_reports(base, [report_path], tmp_path / "merged", tmp_path / "quality")
    assert summary["verified_optional_libraries"] == ["xgboost"]
    assert summary["missing_optional_reports"] == ["catboost", "lightgbm", "onnx", "tensorflow", "torch"]


def test_tampered_report_is_rejected(tmp_path: Path) -> None:
    payload = _runtime_report()
    payload["sample_size"] = 65
    assert "artifact_sha256 does not match" in "\n".join(validate_runtime_report(payload))
    base = _base_evidence(tmp_path)
    report_path = tmp_path / "adapter_report_xgboost_py311.json"
    _write_json(report_path, payload)
    with pytest.raises(ValueError, match="artifact_sha256"):
        merge_reports(base, [report_path], tmp_path / "merged", tmp_path / "quality")


def test_cross_python_status_conflict_is_explicit(tmp_path: Path) -> None:
    base = _base_evidence(tmp_path)
    first = tmp_path / "adapter_report_xgboost_py311.json"
    second = tmp_path / "adapter_report_xgboost_py312.json"
    _write_json(first, _runtime_report(python_version="3.11.9"))
    _write_json(second, _runtime_report(python_version="3.12.4", status="failed"))
    merge_reports(base, [first, second], tmp_path / "merged", tmp_path / "quality")
    matrix = json.loads((tmp_path / "merged/model_support_matrix.json").read_text(encoding="utf-8"))
    row = next(item for item in matrix["configurations"] if item["library"] == "xgboost")
    assert row["status"] == "failed_version_inconsistency"
