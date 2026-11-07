"""Native FuzzyXAI adapter for a real interval type-2 fuzzy-logit model.

The model's FOU changes the point prediction through the average of endpoint
probabilities. The output interval is NOT a statistically calibrated confidence
interval and must be checked by the runtime's conservative IT2 policy gate.
"""
from __future__ import annotations

from dataclasses import replace
from typing import Any, Mapping

import numpy as np

from fuzzyxai.adapters.contracts_v2 import (
    EvidenceChannelDescriptor, ExplanationContext, GlobalModelEvidence,
    LocalModelEvidence, ModelCapabilities, ModelOutputSchema, TaskType,
)
from fuzzyxai.adapters.model import ModelPrediction
from fuzzyxai.adapters.model_v2 import ModelAdapterV2
from fuzzyxai.interval_type2 import IT2FuzzyLogit, content_sha256


class IntervalType2Adapter(ModelAdapterV2):
    """First-class, non-surrogate adapter for coupled IT2 logistic rules."""

    adapter_id = "interval_type2_v2"
    model_family = "native_interval_type2_fuzzy_logit"

    def __init__(self, model: IT2FuzzyLogit, *, task: str | TaskType = "auto",
                 output_decoder: Any = None) -> None:
        if not isinstance(model, IT2FuzzyLogit):
            raise TypeError("IntervalType2Adapter requires an IT2FuzzyLogit")
        if task not in ("auto", "classification", "binary", "binary_classification",
                        TaskType.BINARY_CLASSIFICATION):
            raise ValueError("IT2FuzzyLogit is a binary classifier, not a general model")
        super().__init__(model, task=TaskType.BINARY_CLASSIFICATION, output_decoder=output_decoder)

    def feature_names(self) -> list[str]:
        names = [f"feature_{i}" for i in range(self.model.n_input_features)]
        for feature in self.model.features:
            names[feature.feature_index] = feature.feature_name
        return names

    def model_fingerprint(self) -> str:
        """Hash actual fitted coefficients and fuzzy membership parameters."""
        return content_sha256(self.model.to_dict())

    def _rows(self, inputs: Any) -> list[list[float]]:
        raw = inputs.to_numpy() if hasattr(inputs, "to_numpy") else inputs
        values = np.asarray(raw, dtype=float)
        if values.ndim == 1:
            values = values.reshape(1, -1)
        if values.ndim != 2 or values.shape[0] == 0 or values.shape[1] != self.model.n_input_features:
            raise ValueError("IT2 inputs must be a non-empty 2D batch with the declared feature schema")
        if not np.isfinite(values).all():
            raise ValueError("IT2 inputs contain non-finite observations")
        return values.tolist()

    def predict(self, inputs: Any) -> ModelPrediction:
        rows = self._rows(inputs)
        inferences = [self.model.infer(row) for row in rows]
        # This is a type-2 endpoint-midpoint inference policy, not the type-1
        # nominal probability and not a calibrated predictive probability.
        points = [float(item["probability_midpoint"]) for item in inferences]
        predicted = [int(p >= 0.5) for p in points]
        if self.output_decoder is not None:
            predicted = self.output_decoder(predicted)
        return ModelPrediction(
            predictions=predicted,
            probabilities=[[1.0 - p, p] for p in points],
            model_type="IT2FuzzyLogit",
            adapter_id=self.adapter_id,
            metadata={
                "task_type": TaskType.BINARY_CLASSIFICATION.value,
                "classes": [0, 1],
                "calibrated_probability": False,
                "score_semantics": "type2_parameter_endpoint_midpoint_uncalibrated",
                "source_fit": self.model.source_fit,
                "model_sha256": self.model_fingerprint(),
                "it2_inference": inferences,
                "limitations": [
                    "IT2 FOU measures uncertainty in membership parameters, not prediction coverage",
                    "Endpoint-midpoint predictions are uncalibrated and must pass the runtime IT2 gate",
                ],
            },
        )

    def output_schema(self) -> ModelOutputSchema:
        return ModelOutputSchema(
            task_type=TaskType.BINARY_CLASSIFICATION,
            classes=(0, 1),
            score_semantics="it2_fou_endpoint_midpoint_uncalibrated",
            calibrated_probability=False,
        )

    def capabilities(self) -> ModelCapabilities:
        base = super().capabilities()
        descriptors = (
            EvidenceChannelDescriptor("prediction", True, "native", "IT2FuzzyLogit.infer"),
            EvidenceChannelDescriptor("native_rules", True, "native", "IT2FuzzyLogit.features"),
            EvidenceChannelDescriptor("rule_activation", True, "native", "IT2FuzzyLogit nominal fuzzy memberships"),
            EvidenceChannelDescriptor("it2_fou", True, "native", "IT2FuzzyLogit analytic logit extrema",
                limitations=("FOU output range is not a calibrated prediction interval.",)),
            EvidenceChannelDescriptor("model_fingerprint", True, "native", "SHA256 of full model manifest"),
        )
        return replace(
            base, predict_proba=True, calibration=False, uncertainty=True,
            native_rules=True, rules=True, class_names=True, feature_names=True,
            local_contributions=False, channels=descriptors,
        )

    def extract_rules(self) -> list[Mapping[str, Any]]:
        result: list[Mapping[str, Any]] = []
        for feature in self.model.features:
            for term, coefficient in (("low", feature.low_coefficient), ("high", feature.high_coefficient)):
                result.append({
                    "rule_id": f"it2:{feature.feature_index}:{term}",
                    "antecedents": [f"{feature.feature_name} is {term}"],
                    "consequent": f"add {coefficient:.10g} to model logit",
                    "human_text": f"{feature.feature_name} is {term}: logit term {coefficient:.10g}",
                    "evidence_refs": [f"model_manifest:{self.model_fingerprint()}"],
                    "complexity": 1.0,
                })
        return result

    def extract_local_evidence(self, inputs: Any, prediction: ModelPrediction,
                               context: ExplanationContext) -> LocalModelEvidence:
        del context
        rows = self._rows(inputs)
        inferences = prediction.metadata.get("it2_inference")
        if not isinstance(inferences, list) or len(inferences) != len(rows):
            raise ValueError("IT2 prediction and local evidence are not aligned")
        features = inferences[0]["feature_evidence"]
        activations: list[dict[str, Any]] = []
        for feature in features:
            strength_high = float(feature["high_term_nominal"])
            for term, strength in (("low", 1.0 - strength_high), ("high", strength_high)):
                activations.append({
                    "rule_id": f"it2:{feature['feature_index']}:{term}",
                    "terms": [{
                        "feature": feature["feature_name"], "term": term,
                        "membership_degree": strength,
                        "feature_value": feature["observed_value"],
                    }],
                    "activation_strength": strength,
                    "conclusion": f"logit term ({term}), coefficient disclosed in model manifest",
                    "limitations": [
                        "Nominal firing strength; the full IT2 FOU is recorded separately.",
                    ],
                })
        return LocalModelEvidence(
            channels={
                "activated_rules": activations,
                "it2_feature_evidence": features,
                "it2_inference": inferences,
                "it2_model_sha256": self.model_fingerprint(),
                "contribution_method": "native_coupled_it2_rules_no_posthoc_probability_attribution",
            },
            descriptors=self.capabilities().channels,
            missing_channels=("training_history",),
            limitations=(
                "No additive class-probability attribution is implied by fuzzy logit coefficients.",
                "Interval width describes declared membership-parameter uncertainty, not calibration.",
                "For batches, the local rule listing describes the first explained object only.",
            ),
        )

    def extract_global_evidence(self, context: ExplanationContext) -> GlobalModelEvidence:
        del context
        return GlobalModelEvidence(
            channels={"rule_manifest": self.model.to_dict(), "model_sha256": self.model_fingerprint()},
            descriptors=self.capabilities().channels,
            limitations=("This is a native model, not a post-hoc approximation or a clinical rule set.",),
        )

    def validate(self):
        from fuzzyxai.adapter_conformance import run_adapter_conformance

        sample = np.zeros((2, self.model.n_input_features), dtype=float)
        return run_adapter_conformance(self, sample_batch=sample)
