from .medical_image_to_explanation import MedicalImageToExplanationAdapter
from .tabular_to_explanation import TabularToExplanationAdapter
from .text_to_explanation import TextToExplanationAdapter
from .tabular_classification import TabularClassificationAdapter
from .mlflow_tabular import MlflowTabularAdapter

from .base import BaseAdapter, ValidationResult
from .model import (
    AdapterCapabilities,
    CallableAdapter,
    CustomAdapter,
    ModelAdapter,
    ModelPrediction,
    NativeRuleAdapter,
    PredictProbaAdapter,
    SklearnAdapter,
    resolve_model_adapter,
)
from .contracts_v2 import (
    AdapterConformanceReport,
    AdapterResolutionReport,
    EvidenceChannelDescriptor,
    ExplanationContext,
    ExplanationPlanDecision,
    GlobalModelEvidence,
    LocalModelEvidence,
    ModelCapabilities,
    ModelInputSchema,
    ModelOutputSchema,
    TaskType,
)
from .model_registry import AdapterRegistry, MODEL_ADAPTER_REGISTRY, resolve_model_adapter_v2
from .model_v2 import (
    CallableAdapterV2,
    DecisionFunctionAdapter,
    ModelAdapterV2,
    NativeRuleAdapterV2,
    PredictProbaAdapterV2,
)
from .sklearn_v2 import (
    SklearnEnsembleAdapter,
    SklearnKNNAdapter,
    SklearnLinearAdapter,
    SklearnModelAdapterV2,
    SklearnNaiveBayesAdapter,
    SklearnPipelineAdapter,
    SklearnRegressorAdapter,
    SklearnSVMAdapter,
    SklearnTreeAdapter,
    resolve_sklearn_adapter,
)
from .registry import get_adapter, list_adapters

__all__ = [
    "BaseAdapter",
    "ValidationResult",
    "ModelAdapter",
    "AdapterCapabilities",
    "ModelPrediction",
    "CallableAdapter",
    "PredictProbaAdapter",
    "SklearnAdapter",
    "NativeRuleAdapter",
    "CustomAdapter",
    "resolve_model_adapter",
    "MedicalImageToExplanationAdapter",
    "TabularToExplanationAdapter",
    "TextToExplanationAdapter",
    "TabularClassificationAdapter",
    "MlflowTabularAdapter",
    "get_adapter",
    "list_adapters",
    "TaskType",
    "EvidenceChannelDescriptor",
    "ModelCapabilities",
    "ModelInputSchema",
    "ModelOutputSchema",
    "ExplanationContext",
    "LocalModelEvidence",
    "GlobalModelEvidence",
    "AdapterResolutionReport",
    "AdapterConformanceReport",
    "ExplanationPlanDecision",
    "ModelAdapterV2",
    "CallableAdapterV2",
    "PredictProbaAdapterV2",
    "DecisionFunctionAdapter",
    "NativeRuleAdapterV2",
    "AdapterRegistry",
    "MODEL_ADAPTER_REGISTRY",
    "resolve_model_adapter_v2",
    "SklearnModelAdapterV2",
    "SklearnLinearAdapter",
    "SklearnTreeAdapter",
    "SklearnEnsembleAdapter",
    "SklearnSVMAdapter",
    "SklearnKNNAdapter",
    "SklearnNaiveBayesAdapter",
    "SklearnRegressorAdapter",
    "SklearnPipelineAdapter",
    "resolve_sklearn_adapter",
]
