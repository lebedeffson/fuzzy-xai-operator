from __future__ import annotations

import json
import tempfile
from dataclasses import asdict
from pathlib import Path
from typing import Any

from .contracts import VerticalRun

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
