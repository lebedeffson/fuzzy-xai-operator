from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np

from .contracts_v2 import EvidenceChannelDescriptor, ExplanationContext, LocalModelEvidence, ModelCapabilities, TaskType
from .model import ModelPrediction, _serializable
from .model_v2 import ModelAdapterV2
from .sklearn_v2 import SklearnModelAdapterV2, _feature_names, _rows


def _missing_extra(extra: str, package: str) -> ImportError:
    return ImportError(f"{package} integration requires the optional extra: pip install 'fuzzyxai-operator[{extra}]'")


class _NativeBoostingAdapter(SklearnModelAdapterV2):
    contribution_method = "native_library_contributions"

    def capabilities(self) -> ModelCapabilities:
        base = super().capabilities()
        return replace(
            base,
            local_contributions=True,
            global_importance=True,
            native_rules=True,
            decision_path=True,
            channels=(
                *base.channels,
                EvidenceChannelDescriptor("local_contributions", True, "native", self.contribution_method),
                EvidenceChannelDescriptor("native_rules", True, "native", "library tree dump"),
                EvidenceChannelDescriptor("leaf_indices", True, "native", "library leaf prediction"),
            ),
        )


class XGBoostAdapter(_NativeBoostingAdapter):
    adapter_id = "xgboost_v2"
    model_family = "xgboost"
    contribution_method = "xgboost_pred_contribs"

    def extract_local_evidence(self, inputs: Any, prediction: ModelPrediction, context: ExplanationContext) -> LocalModelEvidence:
        del prediction
        try:
            import xgboost as xgb
        except ImportError as exc:
            raise _missing_extra("xgboost", "XGBoost") from exc
        rows = _rows(inputs)
        names = list(context.feature_names) or _feature_names(self.model, rows.shape[1])
        booster = self.model.get_booster() if hasattr(self.model, "get_booster") else self.model
        matrix = xgb.DMatrix(rows, feature_names=names)
        raw = np.asarray(booster.predict(matrix, pred_contribs=True))
        vector = raw[0, 0] if raw.ndim == 3 else raw[0]
        contributions = {name: float(value) for name, value in zip(names, vector[:-1])}
        reconstruction = float(np.sum(vector))
        return LocalModelEvidence(
            channels={
                "contributions": contributions,
                "base_value": float(vector[-1]),
                "contribution_reconstruction": reconstruction,
                "leaf_indices": _serializable(booster.predict(matrix, pred_leaf=True)),
                "contribution_method": self.contribution_method,
                "library_version": getattr(xgb, "__version__", "unknown"),
            },
            descriptors=(EvidenceChannelDescriptor("local_contributions", True, "native", self.contribution_method),),
            limitations=("Contributions reconstruct the library margin; link transformation may be required for probability.",),
        )


class LightGBMAdapter(_NativeBoostingAdapter):
    adapter_id = "lightgbm_v2"
    model_family = "lightgbm"
    contribution_method = "lightgbm_pred_contrib"

    def extract_local_evidence(self, inputs: Any, prediction: ModelPrediction, context: ExplanationContext) -> LocalModelEvidence:
        del prediction
        try:
            import lightgbm as lgb
        except ImportError as exc:
            raise _missing_extra("lightgbm", "LightGBM") from exc
        rows = _rows(inputs)
        names = list(context.feature_names) or _feature_names(self.model, rows.shape[1])
        raw = np.asarray(self.model.predict(rows, pred_contrib=True))
        vector = raw[0, 0] if raw.ndim == 3 else raw[0]
        leaves = self.model.predict(rows, pred_leaf=True)
        return LocalModelEvidence(
            channels={
                "contributions": {name: float(value) for name, value in zip(names, vector[:-1])},
                "base_value": float(vector[-1]),
                "contribution_reconstruction": float(np.sum(vector)),
                "leaf_indices": _serializable(leaves),
                "contribution_method": self.contribution_method,
                "library_version": getattr(lgb, "__version__", "unknown"),
            },
            descriptors=(EvidenceChannelDescriptor("local_contributions", True, "native", self.contribution_method),),
            limitations=("Contributions reconstruct the library raw score unless documented otherwise.",),
        )


class CatBoostAdapter(_NativeBoostingAdapter):
    adapter_id = "catboost_v2"
    model_family = "catboost"
    contribution_method = "catboost_shap_values"

    def extract_local_evidence(self, inputs: Any, prediction: ModelPrediction, context: ExplanationContext) -> LocalModelEvidence:
        del prediction
        try:
            import catboost
        except ImportError as exc:
            raise _missing_extra("catboost", "CatBoost") from exc
        rows = _rows(inputs)
        names = list(context.feature_names) or _feature_names(self.model, rows.shape[1])
        pool = catboost.Pool(rows, feature_names=names)
        raw = np.asarray(self.model.get_feature_importance(pool, type="ShapValues"))
        vector = raw[0, 0] if raw.ndim == 3 else raw[0]
        leaves = self.model.calc_leaf_indexes(pool)
        return LocalModelEvidence(
            channels={
                "contributions": {name: float(value) for name, value in zip(names, vector[:-1])},
                "base_value": float(vector[-1]),
                "contribution_reconstruction": float(np.sum(vector)),
                "leaf_indices": _serializable(leaves),
                "contribution_method": self.contribution_method,
                "library_version": getattr(catboost, "__version__", "unknown"),
            },
            descriptors=(EvidenceChannelDescriptor("local_contributions", True, "native", self.contribution_method),),
            limitations=("CatBoost SHAP values describe model output, not domain causality.",),
        )


class TorchAdapter(ModelAdapterV2):
    adapter_id = "torch_v2"
    model_family = "pytorch"

    def __init__(self, model: Any, *, task: str | TaskType = "auto", output_decoder: Any = None, **kwargs: Any):
        try:
            import torch
        except ImportError as exc:
            raise _missing_extra("torch", "PyTorch") from exc
        if not isinstance(model, torch.nn.Module):
            raise TypeError("TorchAdapter requires torch.nn.Module")
        super().__init__(model, task=task, output_decoder=output_decoder)
        self.forward_fn = kwargs.get("forward_fn")
        self.input_transform = kwargs.get("input_transform")
        self.target_layer = kwargs.get("target_layer")
        self.ig_steps = int(kwargs.get("ig_steps", 64))
        if self.ig_steps < 1:
            raise ValueError("ig_steps must be a positive number of integration intervals")

    def _tensor(self, inputs: Any, *, gradients: bool = False) -> Any:
        import torch

        tensor = self.input_transform(inputs) if self.input_transform else torch.as_tensor(inputs, dtype=torch.float32)
        return tensor.detach().clone().requires_grad_(gradients)

    def _forward(self, tensor: Any) -> Any:
        return self.forward_fn(self.model, tensor) if self.forward_fn else self.model(tensor)

    def predict(self, inputs: Any) -> ModelPrediction:
        import torch

        was_training = bool(self.model.training)
        self.model.eval()
        try:
            with torch.no_grad():
                output = self._forward(self._tensor(inputs))
            array = output.detach().cpu().numpy()
            if self.task_type == TaskType.REGRESSION:
                predictions, probabilities = array.reshape(-1), None
            else:
                probabilities_tensor = torch.sigmoid(output) if output.shape[-1] == 1 else torch.softmax(output, dim=-1)
                probabilities = probabilities_tensor.detach().cpu().numpy()
                predictions = (probabilities.reshape(-1) >= 0.5).astype(int) if output.shape[-1] == 1 else np.argmax(probabilities, axis=-1)
            if self.output_decoder:
                predictions = self.output_decoder(predictions)
            return ModelPrediction(
                predictions=_serializable(predictions),
                probabilities=_serializable(probabilities),
                model_type=type(self.model).__name__,
                adapter_id=self.adapter_id,
                metadata={"task_type": self.task_type.value, "device": str(next(self.model.parameters()).device), "score_semantics": "probability" if probabilities is not None else "prediction"},
            )
        finally:
            self.model.train(was_training)

    def capabilities(self) -> ModelCapabilities:
        base = super().capabilities()
        return replace(
            base,
            predict_proba=self.task_type != TaskType.REGRESSION,
            gradients=True,
            integrated_gradients=True,
            occlusion=True,
            local_contributions=True,
            channels=(
                *base.channels,
                EvidenceChannelDescriptor("gradients", True, "derived_from_native", "autograd"),
                EvidenceChannelDescriptor("integrated_gradients", True, "derived_from_native", f"{self.ig_steps}-interval trapezoidal zero-baseline integration"),
            ),
        )

    def extract_local_evidence(self, inputs: Any, prediction: ModelPrediction, context: ExplanationContext) -> LocalModelEvidence:
        import torch

        was_training = bool(self.model.training)
        self.model.eval()
        try:
            source = self._tensor(inputs)
            baseline = torch.zeros_like(source)
            with torch.no_grad():
                source_output = self._forward(source)
                baseline_output = self._forward(baseline)
                if source_output.ndim > 1 and source_output.shape[-1] > 1:
                    # The explained output coordinate is fixed for the whole
                    # path. Re-selecting argmax at every alpha integrates a
                    # different piecewise function and invalidates IG
                    # completeness when the winning class changes en route.
                    target_index = int(context.target) if context.target is not None else int(torch.argmax(source_output[0]).item())
                    f_source = float(source_output[0, target_index].item())
                    f_baseline = float(baseline_output[0, target_index].item())
                else:
                    target_index = 0
                    f_source = float(source_output.reshape(-1)[0].item())
                    f_baseline = float(baseline_output.reshape(-1)[0].item())
            gradients = []
            for alpha in torch.linspace(0.0, 1.0, self.ig_steps + 1, device=source.device):
                point = (baseline + alpha * (source - baseline)).detach().requires_grad_(True)
                self.model.zero_grad(set_to_none=True)
                output = self._forward(point)
                if output.ndim > 1 and output.shape[-1] > 1:
                    scalar = output[:, target_index].sum()
                else:
                    scalar = output.sum()
                gradient = torch.autograd.grad(scalar, point, retain_graph=False, create_graph=False)[0]
                gradients.append(gradient.detach())
            stacked = torch.stack(gradients)
            average_gradient = (stacked[0] + stacked[-1] + 2.0 * stacked[1:-1].sum(dim=0)) / (2.0 * (len(gradients) - 1))
            integrated = (source - baseline) * average_gradient
            vector = integrated.detach().cpu().numpy().reshape(-1)
            attribution_sum = float(integrated.detach().sum().item())
            output_delta = f_source - f_baseline
            residual = abs(output_delta - attribution_sum)
            names = list(context.feature_names) or [f"feature_{index}" for index in range(len(vector))]
            return LocalModelEvidence(
                channels={
                    "contributions": {name: float(value) for name, value in zip(names, vector)},
                    "integrated_gradients": _serializable(integrated.detach().cpu().numpy()),
                    "contribution_method": "derived_native_integrated_gradients",
                    "gradient_sanity": bool(np.isfinite(vector).all()),
                    "ig_completeness": {
                        "status": "measured", "target_class": target_index,
                        "baseline": "all-zero input tensor", "F_target_x": f_source,
                        "F_target_baseline": f_baseline, "output_space": "logit",
                        "input_output_delta": output_delta, "attribution_sum": attribution_sum,
                        "completeness_residual": residual,
                        "completeness_relative_error": residual / max(abs(output_delta), 1e-12),
                        "n_steps": self.ig_steps,
                        "integration_points": self.ig_steps + 1,
                        "integration_method": "trapezoidal",
                        "endpoint_handling": "both endpoints included with half weight",
                        "formula": "abs((F_target(x)-F_target(baseline))-sum(attributions))",
                    },
                    "completeness_error": residual,
                },
                descriptors=(EvidenceChannelDescriptor("integrated_gradients", True, "derived_from_native", "autograd integrated gradients"),),
                limitations=("Integrated gradients depend on the zero baseline and do not establish domain causality.",),
            )
        finally:
            self.model.zero_grad(set_to_none=True)
            self.model.train(was_training)


class KerasAdapter(ModelAdapterV2):
    adapter_id = "keras_v2"
    model_family = "keras"

    def __init__(self, model: Any, *, task: str | TaskType = "auto", output_decoder: Any = None, **kwargs: Any):
        try:
            import tensorflow as tf
        except ImportError as exc:
            raise _missing_extra("tensorflow", "TensorFlow/Keras") from exc
        if not isinstance(model, tf.keras.Model):
            raise TypeError("KerasAdapter requires tf.keras.Model")
        super().__init__(model, task=task, output_decoder=output_decoder)
        self.forward_fn = kwargs.get("forward_fn")

    def predict(self, inputs: Any) -> ModelPrediction:
        import tensorflow as tf

        output = self.forward_fn(self.model, inputs) if self.forward_fn else self.model(tf.convert_to_tensor(inputs), training=False)
        array = np.asarray(output.numpy())
        if self.task_type == TaskType.REGRESSION:
            predictions, probabilities = array.reshape(-1), None
        else:
            probabilities = tf.math.sigmoid(output).numpy() if array.shape[-1] == 1 else tf.nn.softmax(output, axis=-1).numpy()
            predictions = (probabilities.reshape(-1) >= 0.5).astype(int) if array.shape[-1] == 1 else np.argmax(probabilities, axis=-1)
        return ModelPrediction(_serializable(predictions), _serializable(probabilities), type(self.model).__name__, self.adapter_id, {"task_type": self.task_type.value})

    def conformance_prediction_reference(self, inputs: Any) -> Any:
        """Canonicalize raw Keras outputs for an independent parity check."""

        import tensorflow as tf

        output = self.forward_fn(self.model, inputs) if self.forward_fn else self.model(tf.convert_to_tensor(inputs), training=False)
        array = np.asarray(output.numpy())
        if self.task_type == TaskType.REGRESSION:
            return array.reshape(-1)
        probabilities = tf.math.sigmoid(output).numpy() if array.shape[-1] == 1 else tf.nn.softmax(output, axis=-1).numpy()
        return (probabilities.reshape(-1) >= 0.5).astype(int) if array.shape[-1] == 1 else np.argmax(probabilities, axis=-1)

    def capabilities(self) -> ModelCapabilities:
        base = super().capabilities()
        return replace(
            base,
            gradients=True,
            integrated_gradients=True,
            local_contributions=True,
            channels=(*base.channels, EvidenceChannelDescriptor("gradients", True, "derived_from_native", "TensorFlow GradientTape")),
        )

    def extract_local_evidence(self, inputs: Any, prediction: ModelPrediction, context: ExplanationContext) -> LocalModelEvidence:
        del prediction
        import tensorflow as tf

        tensor = tf.convert_to_tensor(inputs, dtype=tf.float32)
        with tf.GradientTape() as tape:
            tape.watch(tensor)
            output = self.model(tensor, training=False)
            scalar = tf.reduce_sum(output[:, int(context.target or 0)] if len(output.shape) > 1 and output.shape[-1] > 1 else output)
        gradient = tape.gradient(scalar, tensor)
        values = np.asarray((gradient * tensor).numpy()).reshape(-1)
        names = list(context.feature_names) or [f"feature_{index}" for index in range(len(values))]
        return LocalModelEvidence(
            channels={"contributions": {name: float(value) for name, value in zip(names, values)}, "contribution_method": "derived_native_input_gradient"},
            descriptors=(EvidenceChannelDescriptor("gradients", True, "derived_from_native", "TensorFlow GradientTape"),),
            limitations=("Input gradients are local sensitivity, not causal effects.",),
        )


class ONNXRuntimeAdapter(ModelAdapterV2):
    adapter_id = "onnxruntime_v2"
    model_family = "onnxruntime"

    def __init__(self, model: Any, *, task: str | TaskType = "auto", output_decoder: Any = None, **kwargs: Any):
        try:
            import onnxruntime as ort
        except ImportError as exc:
            raise _missing_extra("onnx", "ONNX Runtime") from exc
        session = ort.InferenceSession(str(Path(model))) if isinstance(model, (str, Path)) else model
        super().__init__(session, task=task, output_decoder=output_decoder)
        self.input_name = kwargs.get("input_name") or session.get_inputs()[0].name
        self.output_names = kwargs.get("output_names") or [item.name for item in session.get_outputs()]

    def predict(self, inputs: Any) -> ModelPrediction:
        rows = np.asarray(inputs, dtype=np.float32)
        outputs = self.model.run(self.output_names, {self.input_name: rows})
        primary = np.asarray(outputs[0])
        if self.task_type == TaskType.REGRESSION:
            predictions, probabilities = primary.reshape(-1), None
        else:
            probabilities = np.asarray(outputs[1]) if len(outputs) > 1 else primary
            predictions = np.argmax(probabilities, axis=-1) if probabilities.ndim > 1 and probabilities.shape[-1] > 1 else (probabilities.reshape(-1) >= 0.5).astype(int)
        return ModelPrediction(_serializable(predictions), _serializable(probabilities), "ONNX InferenceSession", self.adapter_id, {"task_type": self.task_type.value, "output_names": self.output_names})

    def capabilities(self) -> ModelCapabilities:
        base = super().capabilities()
        return replace(
            base,
            predict_proba=self.task_type != TaskType.REGRESSION,
            gradients=False,
            integrated_gradients=False,
            channels=(
                *base.channels,
                EvidenceChannelDescriptor("onnx_outputs", True, "native", "InferenceSession.run"),
                EvidenceChannelDescriptor("gradients", False, "derived", "not exported", limitations=("ONNX gradients are unavailable unless explicitly exported.",)),
            ),
        )

    def extract_local_evidence(self, inputs: Any, prediction: ModelPrediction, context: ExplanationContext) -> LocalModelEvidence:
        del inputs, prediction, context
        return LocalModelEvidence(
            channels={"contribution_method": None},
            descriptors=(EvidenceChannelDescriptor("gradients", False, "derived", "not exported"),),
            missing_channels=("gradients", "integrated_gradients", "native_rules"),
            limitations=("ONNX Runtime exposes prediction outputs only unless intermediate outputs were exported.",),
        )
