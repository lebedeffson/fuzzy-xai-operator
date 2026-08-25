"""Wrapping a model FuzzyXAI has no built-in adapter for.

    python examples/06_custom_model_adapter.py

Any model — not just sklearn/XGBoost/PyTorch/etc. — can be explained by
implementing a small ``ModelAdapterV2`` subclass: only ``predict`` is
required; every other channel (local contributions, rules, global evidence)
defaults to an honest "not available" disclosure unless you implement it.
Here we wrap a trivial hand-written threshold rule that isn't any sklearn
estimator at all.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from fuzzyxai import FuzzyXAI
from fuzzyxai.adapters.contracts_v2 import ExplanationContext, LocalModelEvidence
from fuzzyxai.adapters.model import ModelPrediction
from fuzzyxai.adapters.model_v2 import ModelAdapterV2


class ThresholdModel:
    """A model that isn't sklearn/XGBoost/PyTorch/etc. — just a plain Python object."""

    def __init__(self, threshold: float = 0.5) -> None:
        self.threshold = threshold

    def predict(self, X: np.ndarray) -> np.ndarray:
        return (np.asarray(X)[:, 0] > self.threshold).astype(int)


class ThresholdAdapter(ModelAdapterV2):
    """The only required method is predict(); local evidence is opt-in."""

    adapter_id = "example_threshold_adapter"
    model_family = "custom_threshold_rule"

    def predict(self, inputs: Any) -> ModelPrediction:
        rows = np.atleast_2d(np.asarray(inputs, dtype=float))
        predictions = self.model.predict(rows)
        return ModelPrediction(
            predictions=predictions.tolist(),
            probabilities=None,
            model_type="ThresholdModel",
            adapter_id=self.adapter_id,
            metadata={"task_type": self.task_type.value},
        )

    def extract_local_evidence(self, inputs: Any, prediction: ModelPrediction, context: ExplanationContext) -> LocalModelEvidence:
        del prediction
        rows = np.atleast_2d(np.asarray(inputs, dtype=float))
        names = list(context.feature_names) or [f"feature_{i}" for i in range(rows.shape[1])]
        # The only feature the model actually looks at gets a real,
        # honestly-computed margin; nothing is fabricated for the rest.
        contributions = {names[0]: float(rows[0][0] - self.model.threshold)}
        return LocalModelEvidence(
            channels={"contributions": contributions, "contribution_method": "custom_threshold_margin"},
            limitations=("Only the thresholded feature has a measured contribution; all others are unused by this model.",),
        )


def main() -> None:
    model = ThresholdModel(threshold=0.5)
    adapter = ThresholdAdapter(model, task="classification")

    fx = FuzzyXAI.wrap(model, adapter=adapter)
    result = fx.explain_one([0.8, 0.1, 0.3], feature_names=["signal", "noise_a", "noise_b"])

    print("native prediction:", model.predict(np.array([[0.8, 0.1, 0.3]])))
    print("FuzzyXAI prediction:", result.prediction.predictions)
    print()
    print(result.summary())


if __name__ == "__main__":
    main()
