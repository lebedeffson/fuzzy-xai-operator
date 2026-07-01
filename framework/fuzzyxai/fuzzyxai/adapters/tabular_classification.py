from __future__ import annotations

from typing import Any

from fuzzyxai.adapters.base import BaseAdapter
from fuzzyxai.core.errors import AdapterValidationError
from fuzzyxai.core.types import AdaptedInput


class TabularClassificationAdapter(BaseAdapter):
    repo_id = "external_tabular_model"
    scenario_id = "external_wine_classification"
    required_fields = (
        "scenario_id",
        "source_type",
        "model_name",
        "dataset_name",
        "predicted_class",
        "class_probability",
        "feature_values",
        "feature_importance",
        "quality_metrics",
    )

    def validate(self, payload: dict[str, Any]) -> None:
        super().validate(payload)
        if payload["scenario_id"] != self.scenario_id:
            raise AdapterValidationError(f"expected scenario_id={self.scenario_id}")
        probability = float(payload["class_probability"])
        if not 0.0 <= probability <= 1.0:
            raise AdapterValidationError("class_probability must be in [0, 1]")
        if not isinstance(payload["feature_values"], dict) or not payload["feature_values"]:
            raise AdapterValidationError("feature_values must be a non-empty object")
        if not isinstance(payload["feature_importance"], dict) or not payload["feature_importance"]:
            raise AdapterValidationError("feature_importance must be a non-empty object")

    def to_adapted_input(self, payload: dict[str, Any]) -> AdaptedInput:
        self.validate(payload)
        quality = dict(payload.get("quality_metrics") or {})
        values = {
            "source_type": payload["source_type"],
            "model_name": payload["model_name"],
            "dataset_name": payload["dataset_name"],
            "predicted_class": int(payload["predicted_class"]),
            "class_probability": float(payload["class_probability"]),
            "feature_values": dict(payload["feature_values"]),
            "feature_importance": dict(payload["feature_importance"]),
            "missing_rate": float(quality.get("missing_rate", 0.0)),
            "feature_range_violation": float(quality.get("feature_range_violation", 0.0)),
            "context_values": dict(payload.get("context_values") or {}),
        }
        return AdaptedInput(
            scenario_id=self.scenario_id,
            values=values,
            source=self.repo_id,
            value_sources={key: "external_model_payload" for key in values},
            metadata={
                "source_type": payload["source_type"],
                "external_task": True,
            },
        )
