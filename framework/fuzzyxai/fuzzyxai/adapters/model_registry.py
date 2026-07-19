from __future__ import annotations

from dataclasses import dataclass
from importlib import import_module
from importlib.metadata import entry_points
from pathlib import Path
from typing import Any, Callable, cast

from .contracts_v2 import AdapterResolutionReport, TaskType
from .model import ModelAdapter
from .model_v2 import CallableAdapterV2, DecisionFunctionAdapter, ModelAdapterV2, NativeRuleAdapterV2, PredictProbaAdapterV2
from .sklearn_v2 import resolve_sklearn_adapter


AdapterFactory = Callable[..., ModelAdapter]
AdapterPredicate = Callable[[Any], bool]


@dataclass(frozen=True)
class _Registration:
    name: str
    adapter_factory: AdapterFactory | str
    predicate: AdapterPredicate
    priority: int


def _load_factory(value: AdapterFactory | str) -> AdapterFactory:
    if not isinstance(value, str):
        return value
    module_name, separator, attribute = value.partition(":")
    if not separator:
        raise ValueError(f"invalid lazy adapter reference: {value}")
    return cast(AdapterFactory, getattr(import_module(module_name), attribute))


def _module_starts(model: Any, prefix: str) -> bool:
    return type(model).__module__.startswith(prefix)


class AdapterRegistry:
    """Priority registry with lazy optional integrations and auditable resolution."""

    def __init__(self, *, discover_plugins: bool = True):
        self._registrations: list[_Registration] = []
        self._plugins_discovered = False
        self._discover_plugins_enabled = discover_plugins
        self.last_report: AdapterResolutionReport | None = None

    def register(
        self,
        *,
        adapter_class: AdapterFactory | str,
        predicate: AdapterPredicate,
        priority: int = 0,
        name: str | None = None,
    ) -> None:
        registration_name = name or (adapter_class if isinstance(adapter_class, str) else adapter_class.__name__)
        self._registrations = [item for item in self._registrations if item.name != registration_name]
        self._registrations.append(_Registration(registration_name, adapter_class, predicate, priority))
        self._registrations.sort(key=lambda item: (-item.priority, item.name))

    def discover_entry_points(self) -> None:
        if self._plugins_discovered or not self._discover_plugins_enabled:
            return
        self._plugins_discovered = True
        for point in entry_points(group="fuzzyxai.adapters"):
            loaded = point.load()
            if isinstance(loaded, tuple) and len(loaded) >= 2:
                adapter_class, predicate = loaded[:2]
                priority = int(loaded[2]) if len(loaded) > 2 else 100
            else:
                adapter_class = loaded
                predicate = getattr(loaded, "supports_model", None)
                priority = int(getattr(loaded, "registry_priority", 100))
            if not callable(predicate):
                raise TypeError(f"adapter entry point {point.name!r} must expose supports_model(model)")
            self.register(adapter_class=adapter_class, predicate=predicate, priority=priority, name=f"plugin:{point.name}")

    def registrations(self) -> tuple[str, ...]:
        self.discover_entry_points()
        return tuple(item.name for item in self._registrations)

    def resolve_with_report(
        self,
        model: Any,
        *,
        task: str | TaskType = "auto",
        explicit: str | ModelAdapter = "auto",
        output_decoder: Callable[[Any], Any] | None = None,
    ) -> tuple[ModelAdapter, AdapterResolutionReport]:
        if isinstance(explicit, ModelAdapter):
            task_type = getattr(explicit, "task_type", TaskType.BINARY_CLASSIFICATION)
            report = AdapterResolutionReport(
                selected_adapter=explicit.adapter_id,
                selected_family=str(getattr(explicit, "model_family", "legacy_custom")),
                task_type=task_type,
                matched_predicates=("explicit_instance",),
                rejected_adapters=(),
            )
            self.last_report = report
            return explicit, report
        self.discover_entry_points()
        candidates = self._registrations
        if explicit != "auto":
            aliases = {
                "sklearn": "sklearn_exact",
                "anfis": "native_rules_v2",
                "native_rules": "native_rules_v2",
                "callable": "callable_v2",
                "predict_proba": "predict_proba_v2",
                "decision_function": "decision_function_v2",
            }
            requested = aliases.get(explicit, explicit)
            candidates = [item for item in candidates if item.name == requested]
            if not candidates:
                raise ValueError(f"unknown model adapter: {explicit}")
            selected = candidates[0]
            factory = _load_factory(selected.adapter_factory)
            adapter = factory(model, task=task, output_decoder=output_decoder)
            task_type = getattr(adapter, "task_type", TaskType.BINARY_CLASSIFICATION)
            report = AdapterResolutionReport(
                selected_adapter=adapter.adapter_id,
                selected_family=str(getattr(adapter, "model_family", selected.name)),
                task_type=task_type,
                matched_predicates=(f"explicit:{selected.name}",),
                rejected_adapters=(),
            )
            self.last_report = report
            return adapter, report
        matched: list[_Registration] = []
        rejected: list[str] = []
        warnings: list[str] = []
        for registration in candidates:
            try:
                supported = bool(registration.predicate(model))
            except Exception as exc:
                supported = False
                warnings.append(f"{registration.name} predicate failed: {exc}")
            if supported:
                matched.append(registration)
            else:
                rejected.append(registration.name)
        if not matched:
            raise TypeError("no registered adapter supports this model; provide a CustomAdapterV2")
        selected = matched[0]
        factory = _load_factory(selected.adapter_factory)
        adapter = factory(model, task=task, output_decoder=output_decoder)
        task_type = getattr(adapter, "task_type", TaskType.BINARY_CLASSIFICATION)
        report = AdapterResolutionReport(
            selected_adapter=adapter.adapter_id,
            selected_family=str(getattr(adapter, "model_family", selected.name)),
            task_type=task_type,
            matched_predicates=tuple(item.name for item in matched),
            rejected_adapters=tuple(rejected),
            warnings=tuple(warnings),
        )
        self.last_report = report
        return adapter, report

    def resolve(
        self,
        model: Any,
        *,
        task: str | TaskType = "auto",
        explicit: str | ModelAdapter = "auto",
        output_decoder: Callable[[Any], Any] | None = None,
    ) -> ModelAdapter:
        return self.resolve_with_report(model, task=task, explicit=explicit, output_decoder=output_decoder)[0]


def _sklearn_factory(model: Any, *, task: str | TaskType = "auto", output_decoder: Any = None) -> ModelAdapterV2:
    return resolve_sklearn_adapter(model, task=task, output_decoder=output_decoder)


def default_model_registry() -> AdapterRegistry:
    registry = AdapterRegistry()
    registry.register(
        name="native_rules_v2",
        adapter_class=NativeRuleAdapterV2,
        predicate=lambda model: hasattr(model, "rules_") and callable(getattr(model, "predict_proba", None)),
        priority=500,
    )
    registry.register(
        name="xgboost",
        adapter_class="fuzzyxai.adapters.optional_v2:XGBoostAdapter",
        predicate=lambda model: _module_starts(model, "xgboost"),
        priority=450,
    )
    registry.register(
        name="lightgbm",
        adapter_class="fuzzyxai.adapters.optional_v2:LightGBMAdapter",
        predicate=lambda model: _module_starts(model, "lightgbm"),
        priority=450,
    )
    registry.register(
        name="catboost",
        adapter_class="fuzzyxai.adapters.optional_v2:CatBoostAdapter",
        predicate=lambda model: _module_starts(model, "catboost"),
        priority=450,
    )
    registry.register(
        name="torch",
        adapter_class="fuzzyxai.adapters.optional_v2:TorchAdapter",
        predicate=lambda model: _module_starts(model, "torch"),
        priority=440,
    )
    registry.register(
        name="keras",
        adapter_class="fuzzyxai.adapters.optional_v2:KerasAdapter",
        predicate=lambda model: _module_starts(model, "keras") or _module_starts(model, "tensorflow"),
        priority=440,
    )
    registry.register(
        name="onnx",
        adapter_class="fuzzyxai.adapters.optional_v2:ONNXRuntimeAdapter",
        predicate=lambda model: (
            isinstance(model, (str, bytes, Path)) and str(model).lower().endswith(".onnx")
        )
        or _module_starts(model, "onnxruntime"),
        priority=440,
    )
    registry.register(
        name="sklearn_exact",
        adapter_class=_sklearn_factory,
        predicate=lambda model: _module_starts(model, "sklearn") or hasattr(model, "_estimator_type"),
        priority=400,
    )
    registry.register(
        name="predict_proba_v2",
        adapter_class=PredictProbaAdapterV2,
        predicate=lambda model: callable(getattr(model, "predict_proba", None)),
        priority=200,
    )
    registry.register(
        name="decision_function_v2",
        adapter_class=DecisionFunctionAdapter,
        predicate=lambda model: callable(getattr(model, "decision_function", None)),
        priority=150,
    )
    registry.register(
        name="callable_v2",
        adapter_class=CallableAdapterV2,
        predicate=callable,
        priority=100,
    )
    return registry


MODEL_ADAPTER_REGISTRY = default_model_registry()


def resolve_model_adapter_v2(
    model: Any,
    *,
    task: str | TaskType = "auto",
    adapter: str | ModelAdapter = "auto",
    output_decoder: Callable[[Any], Any] | None = None,
) -> tuple[ModelAdapter, AdapterResolutionReport]:
    return MODEL_ADAPTER_REGISTRY.resolve_with_report(
        model,
        task=task,
        explicit=adapter,
        output_decoder=output_decoder,
    )
