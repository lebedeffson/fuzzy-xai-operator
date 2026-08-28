from __future__ import annotations

from dataclasses import replace
from typing import Any

import numpy as np
from fuzzyxai.adapters.model import ModelPrediction
from fuzzyxai.adapters.optional_v2 import TorchAdapter


class TemperatureScaledTorchAdapter(TorchAdapter):
    """Experiment-side probability calibration; native IG remains logit-space."""

    adapter_id = "torch_temperature_scaled_ch6"

    def __init__(self, model: Any, *, temperature: float, **kwargs: Any):
        if not np.isfinite(temperature) or temperature <= 0:
            raise ValueError("temperature must be finite and positive")
        super().__init__(model, **kwargs)
        self.temperature = float(temperature)

    def predict(self, inputs: Any) -> ModelPrediction:
        import torch

        was_training = bool(self.model.training); self.model.eval()
        try:
            with torch.no_grad():
                output = self._forward(self._tensor(inputs))
                probabilities = torch.softmax(output / self.temperature, dim=-1).cpu().numpy()
            predictions = np.argmax(probabilities, axis=-1)
            if self.output_decoder:
                predictions = self.output_decoder(predictions)
            return ModelPrediction(predictions=predictions.tolist(), probabilities=probabilities.tolist(), model_type=type(self.model).__name__, adapter_id=self.adapter_id, metadata={"task_type": self.task_type.value, "device": str(next(self.model.parameters()).device), "score_semantics": "temperature_scaled_probability", "temperature": self.temperature})
        finally:
            self.model.train(was_training)

    def extract_local_evidence(self, inputs: Any, prediction: ModelPrediction, context: Any) -> Any:
        """Keep full tensor IG without duplicating it as thousands of claims.

        The public Torch adapter exposes both the tensor and a flattened
        ``contributions`` mapping.  For medical images and ECG the latter is a
        redundant projection (268k and 12k entries respectively) that makes
        human-report composition quadratic.  The signed tensor, completeness
        trace and method provenance remain unchanged.
        """

        evidence = super().extract_local_evidence(inputs, prediction, context)
        channels = dict(evidence.channels)
        channels.pop("contributions", None)
        channels["flat_contributions_status"] = "not_materialized_redundant_with_full_ig_tensor"
        return replace(evidence, channels=channels)
