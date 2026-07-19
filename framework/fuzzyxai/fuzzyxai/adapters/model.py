from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import asdict, dataclass, field, replace
from hashlib import sha256
from typing import Any, Callable, Mapping, cast


def _serializable(value: Any) -> Any:
    if hasattr(value, "tolist"):
        return value.tolist()
    if isinstance(value, Mapping):
        return {str(key): _serializable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_serializable(item) for item in value]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)


@dataclass(frozen=True)
class ModelPrediction:
    """Canonical, serializable output of an arbitrary predictive model."""

    predictions: Any
    probabilities: Any = None
    model_type: str = "unknown"
    adapter_id: str = "custom"
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return cast(dict[str, Any], _serializable(asdict(self)))

    def primary_score(self) -> float | None:
        values = _serializable(self.probabilities)
        if values is None:
            return None
        while isinstance(values, list) and values:
            if all(isinstance(item, (int, float)) for item in values):
                return float(max(values))
            values = values[0]
        return float(values) if isinstance(values, (int, float)) else None


@dataclass(frozen=True)
class AdapterCapabilities:
    """Typed disclosure of native evidence channels exposed by an adapter."""

    predict: bool = True
    predict_proba: bool = False
    feature_importance: bool = False
    gradients: bool = False
    rules: bool = False
    embeddings: bool = False
    training_history: bool = False
    checkpoints: bool = False

    def get(self, name: str, default: bool = False) -> bool:
        return bool(getattr(self, name, default))

    def to_dict(self) -> dict[str, bool]:
        return {name: bool(value) for name, value in asdict(self).items()}


class ModelAdapter(ABC):
    """Canonical contract between a predictive model and FuzzyXAI."""

    adapter_id = "model"

    def __init__(self, model: Any):
        self.model = model

    @abstractmethod
    def predict(self, inputs: Any) -> ModelPrediction:
        """Run the model and return a canonical prediction artifact."""

    def capabilities(self) -> AdapterCapabilities:
        """Declare evidence channels that the adapter can provide."""

        return AdapterCapabilities()

    def feature_names(self) -> list[str]:
        names = getattr(self.model, "feature_names_in_", None)
        return [str(item) for item in names] if names is not None else []

    def extract_rules(self) -> list[Mapping[str, Any]]:
        """Return native model rules; the default adapter has none."""

        return []

    def extract_internal_evidence(self, inputs: Any) -> Mapping[str, Any]:
        """Return model-native local facts; unavailable channels stay absent."""

        return {}

    def model_fingerprint(self) -> str:
        """Return a structural fingerprint without serializing training data."""

        getter = getattr(self.model, "get_params", None)
        parameters = getter(deep=False) if callable(getter) else {}
        payload = (type(self.model).__module__, type(self.model).__qualname__, repr(sorted(parameters.items())))
        return sha256(repr(payload).encode("utf-8")).hexdigest()


class CallableAdapter(ModelAdapter):
    adapter_id = "callable"

    def __init__(self, model: Callable[[Any], Any]):
        if not callable(model):
            raise TypeError("CallableAdapter requires a callable model")
        super().__init__(model)

    def predict(self, inputs: Any) -> ModelPrediction:
        output = self.model(inputs)
        return ModelPrediction(
            predictions=_serializable(output),
            model_type=type(self.model).__name__,
            adapter_id=self.adapter_id,
        )


class PredictProbaAdapter(ModelAdapter):
    adapter_id = "predict_proba"

    def __init__(self, model: Any):
        if not callable(getattr(model, "predict_proba", None)):
            raise TypeError("PredictProbaAdapter requires model.predict_proba")
        super().__init__(model)

    def predict(self, inputs: Any) -> ModelPrediction:
        probabilities = self.model.predict_proba(inputs)
        predictor = getattr(self.model, "predict", None)
        predictions = predictor(inputs) if callable(predictor) else probabilities
        return ModelPrediction(
            predictions=_serializable(predictions),
            probabilities=_serializable(probabilities),
            model_type=type(self.model).__name__,
            adapter_id=self.adapter_id,
            metadata={"classes": _serializable(getattr(self.model, "classes_", None))},
        )

    def capabilities(self) -> AdapterCapabilities:
        return replace(super().capabilities(), predict_proba=True)


class SklearnAdapter(PredictProbaAdapter):
    """Adapter for fitted scikit-learn compatible classifiers."""

    adapter_id = "sklearn"

    def capabilities(self) -> AdapterCapabilities:
        has_rule_like_structure = any(hasattr(self.model, name) for name in ("tree_", "estimators_", "coef_"))
        return replace(
            super().capabilities(),
            feature_importance=hasattr(self.model, "feature_importances_") or hasattr(self.model, "coef_"),
            rules=has_rule_like_structure,
        )

    def extract_internal_evidence(self, inputs: Any) -> Mapping[str, Any]:
        values = inputs.to_numpy() if hasattr(inputs, "to_numpy") else inputs
        rows = values.tolist() if hasattr(values, "tolist") else list(values)
        first = rows[0] if rows and isinstance(rows[0], (list, tuple)) else rows
        names = self.feature_names() or [f"feature_{index}" for index in range(len(first))]
        coefficients = getattr(self.model, "coef_", None)
        if coefficients is not None:
            coefficient_rows = _serializable(coefficients)
            row = coefficient_rows[0] if coefficient_rows and isinstance(coefficient_rows[0], list) else coefficient_rows
            return {
                "contributions": {name: float(value) * float(weight) for name, value, weight in zip(names, first, row)},
                "contribution_method": "native_linear_term_x_coefficient",
                "limitations": ["linear terms do not include unmodelled interactions"],
            }
        importance = getattr(self.model, "feature_importances_", None)
        if importance is not None:
            return {
                "contributions": {name: float(value) for name, value in zip(names, _serializable(importance))},
                "contribution_method": "global_tree_feature_importance",
                "limitations": ["global feature importance is not a local causal contribution"],
            }
        return {}


class NativeRuleAdapter(PredictProbaAdapter):
    """Adapter for ANFIS/fuzzy models exposing auditable ``rules_`` records."""

    adapter_id = "native_rules"

    def __init__(self, model: Any):
        super().__init__(model)
        if not hasattr(model, "rules_"):
            raise TypeError("NativeRuleAdapter requires model.rules_")

    def capabilities(self) -> AdapterCapabilities:
        return replace(super().capabilities(), rules=True, feature_importance=True)

    def extract_rules(self) -> list[Mapping[str, Any]]:
        return [dict(item) for item in getattr(self.model, "rules_", [])]


class CustomAdapter(ModelAdapter):
    """Template base class for model-specific integrations."""


def resolve_model_adapter(model: Any, adapter: str | ModelAdapter = "auto") -> ModelAdapter:
    """Resolve an explicit adapter or select one from model capabilities."""

    if isinstance(adapter, ModelAdapter):
        return adapter
    if adapter == "auto":
        if hasattr(model, "rules_") and callable(getattr(model, "predict_proba", None)):
            return NativeRuleAdapter(model)
        if callable(getattr(model, "predict_proba", None)):
            module = type(model).__module__
            if module.startswith("sklearn") or any(hasattr(model, name) for name in ("tree_", "estimators_", "coef_")):
                return SklearnAdapter(model)
            return PredictProbaAdapter(model)
        if callable(model):
            return CallableAdapter(model)
        raise TypeError("auto adapter requires predict_proba or a callable model")
    if adapter in {"predict_proba", "sklearn"}:
        return SklearnAdapter(model) if adapter == "sklearn" else PredictProbaAdapter(model)
    if adapter in {"native_rules", "anfis"}:
        return NativeRuleAdapter(model)
    if adapter == "callable":
        return CallableAdapter(model)
    raise ValueError(f"unknown model adapter: {adapter}")
