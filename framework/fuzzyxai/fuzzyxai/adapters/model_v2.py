from __future__ import annotations

import warnings
from abc import abstractmethod
from dataclasses import replace
from typing import Any, Callable, Mapping

import numpy as np

from .contracts_v2 import (
    AdapterConformanceReport,
    EvidenceChannelDescriptor,
    ExplanationContext,
    GlobalModelEvidence,
    LocalModelEvidence,
    ModelCapabilities,
    ModelInputSchema,
    ModelOutputSchema,
    TaskType,
)
from .model import ModelAdapter, ModelPrediction, _serializable


def infer_task_type(model: Any, requested: str | TaskType = "auto") -> TaskType:
    if isinstance(requested, TaskType):
        return requested
    if requested != "auto":
        aliases = {
            "classification": TaskType.BINARY_CLASSIFICATION,
            "binary": TaskType.BINARY_CLASSIFICATION,
            "multiclass": TaskType.MULTICLASS_CLASSIFICATION,
            "regression": TaskType.REGRESSION,
            "anomaly": TaskType.ANOMALY_DETECTION,
            "forecasting": TaskType.FORECASTING,
            "text": TaskType.TEXT_CLASSIFICATION,
            "image": TaskType.IMAGE_CLASSIFICATION,
        }
        if requested in aliases:
            return aliases[requested]
        try:
            return TaskType(requested)
        except ValueError as exc:
            raise ValueError(f"unsupported task type: {requested}") from exc
    estimator_type = getattr(model, "_estimator_type", None)
    if estimator_type == "regressor":
        return TaskType.REGRESSION
    classes = getattr(model, "classes_", None)
    if classes is not None:
        return TaskType.BINARY_CLASSIFICATION if len(classes) == 2 else TaskType.MULTICLASS_CLASSIFICATION
    if callable(getattr(model, "predict_proba", None)):
        return TaskType.BINARY_CLASSIFICATION
    return TaskType.BINARY_CLASSIFICATION


class ModelAdapterV2(ModelAdapter):
    """Capability-based model contract; the v1 adapter remains a compatibility base."""

    adapter_id = "model_v2"
    model_family = "generic"

    def __init__(
        self,
        model: Any,
        *,
        task: str | TaskType = "auto",
        output_decoder: Callable[[Any], Any] | None = None,
    ):
        super().__init__(model)
        self.task_type = infer_task_type(model, task)
        self.output_decoder = output_decoder

    @abstractmethod
    def predict(self, inputs: Any) -> ModelPrediction:
        pass

    def capabilities(self) -> ModelCapabilities:
        return ModelCapabilities(
            predict=True,
            feature_names=bool(self.feature_names()),
            class_names=hasattr(self.model, "classes_"),
            perturbation_counterfactual=True,
            channels=(
                EvidenceChannelDescriptor("prediction", True, "native", "model.predict"),
                EvidenceChannelDescriptor(
                    "perturbation_counterfactual",
                    True,
                    "derived",
                    "model_re_evaluation",
                    limitations=("Perturbation is not a domain-validated intervention.",),
                ),
            ),
        )

    def input_schema(self) -> ModelInputSchema:
        return ModelInputSchema(feature_names=tuple(self.feature_names()))

    def output_schema(self) -> ModelOutputSchema:
        classes = tuple(_serializable(getattr(self.model, "classes_", ())) or ())
        return ModelOutputSchema(
            task_type=self.task_type,
            classes=classes,
            score_semantics="probability" if callable(getattr(self.model, "predict_proba", None)) else "model_output",
            calibrated_probability=bool(getattr(self.model, "probability", False) or callable(getattr(self.model, "predict_proba", None))),
        )

    def extract_local_evidence(
        self,
        inputs: Any,
        prediction: ModelPrediction,
        context: ExplanationContext,
    ) -> LocalModelEvidence:
        del inputs, prediction, context
        return LocalModelEvidence(
            missing_channels=("local_contributions", "native_rules", "training_history"),
            limitations=("The model exposes prediction only; internal evidence is unavailable.",),
        )

    def extract_global_evidence(self, context: ExplanationContext) -> GlobalModelEvidence:
        del context
        return GlobalModelEvidence(limitations=("The model exposes no global evidence channel.",))

    def extract_internal_evidence(self, inputs: Any) -> Mapping[str, Any]:
        prediction = self.predict(inputs)
        context = ExplanationContext(feature_names=tuple(self.feature_names()))
        return self.extract_local_evidence(inputs, prediction, context).to_runtime_mapping()

    def validate(self) -> AdapterConformanceReport:
        from fuzzyxai.adapter_conformance import run_adapter_conformance

        sample_count = max(1, int(getattr(self.model, "n_features_in_", 1)))
        sample: np.ndarray[Any, Any] = np.zeros((1, sample_count), dtype=float)
        return run_adapter_conformance(self, sample_batch=sample)


class CallableAdapterV2(ModelAdapterV2):
    adapter_id = "callable"
    model_family = "callable_black_box"

    def __init__(
        self,
        model: Callable[[Any], Any],
        *,
        task: str | TaskType = "auto",
        output_decoder: Callable[[Any], Any] | None = None,
    ):
        if not callable(model):
            raise TypeError("CallableAdapterV2 requires a callable model")
        super().__init__(model, task=task, output_decoder=output_decoder)

    def predict(self, inputs: Any) -> ModelPrediction:
        output = self.model(inputs)
        decoded = self.output_decoder(output) if self.output_decoder else output
        return ModelPrediction(
            predictions=_serializable(decoded),
            model_type=type(self.model).__name__,
            adapter_id=self.adapter_id,
            metadata={"task_type": self.task_type.value, "score_semantics": "model_output"},
        )


class PredictProbaAdapterV2(ModelAdapterV2):
    adapter_id = "predict_proba"
    model_family = "predict_proba_compatible"

    def __init__(self, model: Any, *, task: str | TaskType = "auto", output_decoder: Callable[[Any], Any] | None = None):
        if not callable(getattr(model, "predict_proba", None)):
            raise TypeError("PredictProbaAdapterV2 requires model.predict_proba")
        super().__init__(model, task=task, output_decoder=output_decoder)

    def predict(self, inputs: Any) -> ModelPrediction:
        probabilities = self.model.predict_proba(inputs)
        predictor = getattr(self.model, "predict", None)
        predictions = predictor(inputs) if callable(predictor) else np.argmax(np.asarray(probabilities), axis=-1)
        if self.output_decoder:
            predictions = self.output_decoder(predictions)
        return ModelPrediction(
            predictions=_serializable(predictions),
            probabilities=_serializable(probabilities),
            model_type=type(self.model).__name__,
            adapter_id=self.adapter_id,
            metadata={
                "classes": _serializable(getattr(self.model, "classes_", None)),
                "task_type": self.task_type.value,
                "score_semantics": "probability",
            },
        )

    def capabilities(self) -> ModelCapabilities:
        base = super().capabilities()
        return replace(
            base,
            predict_proba=True,
            calibration=True,
            channels=(*base.channels, EvidenceChannelDescriptor("predict_proba", True, "native", "model.predict_proba")),
        )


class DecisionFunctionAdapter(ModelAdapterV2):
    adapter_id = "decision_function"
    model_family = "decision_function_compatible"

    def __init__(self, model: Any, *, task: str | TaskType = "auto", output_decoder: Callable[[Any], Any] | None = None):
        if not callable(getattr(model, "decision_function", None)):
            raise TypeError("DecisionFunctionAdapter requires model.decision_function")
        super().__init__(model, task=task, output_decoder=output_decoder)

    def predict(self, inputs: Any) -> ModelPrediction:
        scores = self.model.decision_function(inputs)
        predictor = getattr(self.model, "predict", None)
        predictions = predictor(inputs) if callable(predictor) else (np.asarray(scores) >= 0).astype(int)
        if self.output_decoder:
            predictions = self.output_decoder(predictions)
        return ModelPrediction(
            predictions=_serializable(predictions),
            model_type=type(self.model).__name__,
            adapter_id=self.adapter_id,
            metadata={
                "decision_scores": _serializable(scores),
                "classes": _serializable(getattr(self.model, "classes_", None)),
                "task_type": self.task_type.value,
                "score_semantics": "uncalibrated_decision_score",
                "calibrated_probability": False,
            },
        )

    def capabilities(self) -> ModelCapabilities:
        base = super().capabilities()
        return replace(
            base,
            decision_function=True,
            channels=(
                *base.channels,
                EvidenceChannelDescriptor(
                    "decision_function",
                    True,
                    "native",
                    "model.decision_function",
                    limitations=("Decision score is not a calibrated probability.",),
                ),
            ),
        )


class NativeRuleAdapterV2(PredictProbaAdapterV2):
    adapter_id = "native_rules_v2"
    model_family = "native_rule_model"

    def __init__(self, model: Any, *, task: str | TaskType = "auto", output_decoder: Callable[[Any], Any] | None = None):
        if not hasattr(model, "rules_"):
            raise TypeError("NativeRuleAdapterV2 requires model.rules_")
        super().__init__(model, task=task, output_decoder=output_decoder)

    def capabilities(self) -> ModelCapabilities:
        base = super().capabilities()
        return replace(
            base,
            native_rules=True,
            local_contributions=True,
            channels=(
                *base.channels,
                EvidenceChannelDescriptor("native_rules", True, "native", "model.rules_"),
                EvidenceChannelDescriptor("rule_activation", True, "native", "model.rules_ activation"),
            ),
        )

    def extract_rules(self) -> list[Mapping[str, Any]]:
        return [dict(item) for item in getattr(self.model, "rules_", ())]

    def extract_local_evidence(
        self,
        inputs: Any,
        prediction: ModelPrediction,
        context: ExplanationContext,
    ) -> LocalModelEvidence:
        del inputs, prediction, context
        rules = self.extract_rules()
        activations = {str(rule.get("rule_id", index)): rule.get("activation") for index, rule in enumerate(rules)}
        return LocalModelEvidence(
            channels={"rules": rules, "rule_activations": activations, "contribution_method": "native_rule_activation"},
            descriptors=(
                EvidenceChannelDescriptor("native_rules", True, "native", "model.rules_"),
                EvidenceChannelDescriptor("rule_activation", True, "native", "model.rules_ activation"),
            ),
            missing_channels=("training_history",) if not hasattr(self.model, "history_") else (),
        )


def warn_legacy_adapter() -> None:
    warnings.warn(
        "ModelAdapter v1 is deprecated and will be removed after v1.5.0; implement ModelAdapterV2 instead.",
        DeprecationWarning,
        stacklevel=2,
    )
