from __future__ import annotations

import json
import os
import tempfile
from dataclasses import asdict
from pathlib import Path
from typing import Any

from .contracts import VerticalRun
from .pipeline import V2_SCENARIOS, PipelineRun

ARTIFACTS = {
    "request.json": "request",
    "prediction.json": "prediction",
    "local_explanation.json": "explanation",
    "explainable_object.json": "explainable_object",
    "route_graph.json": "route_graph",
    "diagnostic_report.json": "diagnosis",
    "user_response.json": ("views", "user"),
    "engineer_response.json": ("views", "engineer"),
    "auditor_response.json": ("views", "auditor"),
}


def log_run(run: VerticalRun, *, tracking_uri: str, experiment_name: str = "fuzzyxai-ml-vertical-v1") -> dict[str, Any]:
    """Log a completed run to a real MLflow backend without requiring network."""
    try:
        import mlflow
    except ImportError as exc:  # pragma: no cover - exercised by clean-env acceptance
        raise RuntimeError("MLflow integration requires the locked ml-vertical extra") from exc

    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(experiment_name)
    payload = asdict(run)
    with mlflow.start_run(run_name=run.run_id.replace(":", "-")) as active:
        mlflow.log_params(
            {
                "scenario_id": run.scenario_id,
                "object_id": run.request["object_id"],
                "model_id": (run.prediction or {}).get("model_id", "not_executed"),
                "model_version": (run.prediction or {}).get("model_version", "not_executed"),
                "explainer_id": (run.explanation or {}).get("explainer_id", "not_executed"),
                "explainer_version": (run.explanation or {}).get("explainer_version", "not_executed"),
                "explain_plan_id": "bcw-logreg-shap-v1",
                "explain_plan_version": "1.0.0",
                "representation": run.representation["representation_id"],
                "action": run.observer["action"],
                "route_status": run.diagnosis["route_status"],
            }
        )
        mlflow.log_metrics(
            {
                "prediction": float((run.prediction or {}).get("probability", 0.0)),
                "reduction_loss": float(run.reduction["loss"]),
                "diagnostic_issue_count": float(len(run.diagnosis["issues"])),
            }
        )
        mlflow.set_tags({"fuzzyxai.run_id": run.run_id, "fuzzyxai.canonical_sha256": run.canonical_sha256})
        with tempfile.TemporaryDirectory(prefix="fuzzyxai-mlv1-") as temp:
            root = Path(temp)
            for filename, selector in ARTIFACTS.items():
                value = payload[selector] if isinstance(selector, str) else payload[selector[0]][selector[1]]
                path = root / filename
                path.write_text(json.dumps(value, ensure_ascii=False, indent=2, default=str) + "\n", encoding="utf-8")
                mlflow.log_artifact(str(path), artifact_path="vertical")
        return {
            "status": "MLFLOW_INTEGRATION_PASS",
            "run_id": active.info.run_id,
            "artifact_uri": active.info.artifact_uri,
            "tracking_uri": tracking_uri,
            "artifacts": tuple(ARTIFACTS),
        }


PIPELINE_ARTIFACTS = {
    "canonical_result.json": None,
    "route_graph.json": "route_graph",
    "contract_report.json": "contract_report",
    "diagnosis.json": "diagnosis",
    "repair_plan.json": "repair_plan",
    "recertification.json": "recertification",
    "user_view.json": ("views", "user"),
    "engineering_view.json": ("views", "engineering"),
    "audit_view.json": ("views", "audit"),
    "dataset_manifest.json": ("manifests", "dataset_manifest"),
    "split_manifest.json": ("manifests", "split_manifest"),
    "training_configuration.json": ("manifests", "training_configuration"),
    "model_manifest.json": ("manifests", "model_manifest"),
    "preprocessor_manifest.json": ("manifests", "preprocessor_manifest"),
}


def log_pipeline_run(
    run: PipelineRun,
    *,
    tracking_uri: str,
    experiment_name: str = "fuzzyxai-ml-pipeline-v2",
) -> dict[str, Any]:
    """Log the complete pipeline run and its stage-level evidence to MLflow."""
    try:
        import mlflow
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError("MLflow integration requires the locked ml-vertical extra") from exc

    payload = asdict(run)
    manifests = run.manifests
    expected = V2_SCENARIOS.get(run.scenario_id)
    diagnosis_correct = not expected or (
        run.diagnosis["failed_stage"] == expected["stage"]
        and run.diagnosis["violated_contract"] == expected["contract"]
    )
    action_correct = not expected or run.diagnosis["recommended_action"] == expected["action"]
    critical = sum(item["severity"] == "HIGH" for item in run.contract_report["violations"])
    repaired = bool(run.recertification and run.recertification.get("full_recertification"))
    false_certification = run.diagnosis["recommended_action"] == "ACCEPT" and critical > 0 and not repaired
    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(experiment_name)
    with mlflow.start_run(run_name=run.run_id.replace(":", "-")) as active:
        mlflow.log_params(
            {
                "scenario_id": run.scenario_id,
                "pipeline_version": run.pipeline_version,
                "dataset_id": manifests["dataset_manifest"]["dataset_id"],
                "dataset_sha256": manifests["dataset_manifest"]["dataset_sha256"],
                "split_sha256": manifests["split_manifest"]["split_sha256"],
                "preprocessor_sha256": manifests["preprocessor_manifest"]["artifact_sha256"],
                "model_sha256": manifests["model_manifest"]["model_sha256"],
                "feature_schema_sha256": manifests["model_manifest"]["feature_schema_sha256"],
                "explainer_version": manifests["explainer_manifest"]["version"],
                "git_commit": os.getenv("FUZZYXAI_GIT_COMMIT", "unavailable"),
            }
        )
        mlflow.log_metrics(
            {
                "pipeline_valid": float(run.pipeline_status == "VALID"),
                "critical_violation_count": float(critical),
                "contract_accuracy": float(diagnosis_correct),
                "stage_localization_accuracy": float(diagnosis_correct),
                "observer_action_accuracy": float(action_correct),
                "repair_success": float(repaired),
                "recertification_success": float(repaired),
                "false_certification": float(false_certification),
                "runtime_ms": float(run.runtime_ms),
            }
        )
        mlflow.set_tags(
            {
                "fuzzyxai.run_id": run.run_id,
                "fuzzyxai.canonical_sha256": run.canonical_sha256,
                "fuzzyxai.failed_stage": str(run.diagnosis["failed_stage"]),
                "fuzzyxai.violated_contract": str(run.diagnosis["violated_contract"]),
            }
        )
        with tempfile.TemporaryDirectory(prefix="fuzzyxai-mlp2-") as temp:
            root = Path(temp)
            for filename, selector in PIPELINE_ARTIFACTS.items():
                if selector is None:
                    value = payload
                elif isinstance(selector, str):
                    value = payload[selector]
                else:
                    value = payload[selector[0]][selector[1]]
                path = root / filename
                path.write_text(json.dumps(value, ensure_ascii=False, indent=2, default=str) + "\n", encoding="utf-8")
                mlflow.log_artifact(str(path), artifact_path="pipeline")
        return {
            "status": "PIPELINE_MLFLOW_INTEGRATION_PASS",
            "run_id": active.info.run_id,
            "artifact_uri": active.info.artifact_uri,
            "tracking_uri": tracking_uri,
            "artifacts": tuple(PIPELINE_ARTIFACTS),
        }
