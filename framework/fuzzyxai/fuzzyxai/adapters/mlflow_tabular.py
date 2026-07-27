from __future__ import annotations

from typing import Any

from fuzzyxai.adapters.base import BaseAdapter
from fuzzyxai.core.errors import AdapterValidationError
from fuzzyxai.core.types import AdaptedInput


class MlflowTabularAdapter(BaseAdapter):
    adapter_id = "mlflow_tabular"
    task_type = "tabular_classification"
    supported_payload_schema = "mlflow_registered_tabular_model"
    repo_id = "local_mlflow_tracking_store"
    scenario_id = "mlflow_tabular_classification"
    required_fields = (
        "scenario_id",
        "model_name",
        "dataset_name",
        "predicted_class",
        "class_probability",
        "feature_values",
        "feature_importance",
        "run_id",
        "model_version",
        "artifact_uri",
        "mlflow_version",
        "mlflow_params",
        "mlflow_tags",
    )

    def validate(self, payload: dict[str, Any]) -> None:
        BaseAdapter.validate(self, payload)
        if payload["scenario_id"] != self.scenario_id:
            raise AdapterValidationError(
                f"expected scenario_id={self.scenario_id}"
            )
        probability = float(payload["class_probability"])
        if not 0.0 <= probability <= 1.0:
            raise AdapterValidationError(
                "class_probability must be in [0, 1]"
            )
        for field in ("feature_values", "feature_importance"):
            if not isinstance(payload[field], dict) or not payload[field]:
                raise AdapterValidationError(
                    f"{field} must be a non-empty object"
                )
        for field in ("mlflow_params", "mlflow_tags"):
            if not isinstance(payload[field], dict):
                raise AdapterValidationError(
                    f"{field} must be an object"
                )
        for field in (
            "run_id",
            "model_version",
            "artifact_uri",
            "mlflow_version",
        ):
            if not str(payload[field]).strip():
                raise AdapterValidationError(
                    f"{field} must be registered"
                )

    def to_adapted_input(
        self,
        payload: dict[str, Any],
    ) -> AdaptedInput:
        self.validate(payload)
        quality = dict(payload.get("quality_metrics") or {})
        values = {
            "source_type": "mlflow_model_registry",
            "model_name": str(payload["model_name"]),
            "dataset_name": str(payload["dataset_name"]),
            "predicted_class": int(payload["predicted_class"]),
            "class_probability": float(payload["class_probability"]),
            "feature_values": dict(payload["feature_values"]),
            "feature_importance": dict(payload["feature_importance"]),
            "top_k_importance": int(
                payload.get(
                    "top_k_importance",
                    len(payload["feature_importance"]),
                )
            ),
            "missing_rate": float(quality.get("missing_rate", 0.0)),
            "feature_range_violation": float(
                quality.get("feature_range_violation", 0.0)
            ),
            "context_values": {
                **dict(payload.get("context_values") or {}),
                "task_type": "tabular_classification",
            },
            "run_id": str(payload["run_id"]),
            "model_version": str(payload["model_version"]),
            "artifact_uri": str(payload["artifact_uri"]),
            "mlflow_version": str(payload["mlflow_version"]),
            "mlflow_params": dict(payload["mlflow_params"]),
            "mlflow_tags": dict(payload["mlflow_tags"]),
        }
        return AdaptedInput(
            scenario_id=self.scenario_id,
            values=values,
            source=self.repo_id,
            value_sources={
                key: "mlflow_tracking_and_model_registry"
                for key in values
            },
            metadata={
                "external_task": True,
                "provenance_provider": "MLflow",
                "run_id": values["run_id"],
                "model_version": values["model_version"],
            },
        )
