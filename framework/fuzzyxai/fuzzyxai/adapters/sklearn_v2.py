from __future__ import annotations

from dataclasses import replace
from typing import Any

import numpy as np
from numpy.typing import NDArray

from .contracts_v2 import (
    EvidenceChannelDescriptor,
    ExplanationContext,
    LocalModelEvidence,
    ModelCapabilities,
    ModelInputSchema,
    EvidenceOrigin,
    TaskType,
)
from .model import ModelPrediction, _serializable
from .model_v2 import ModelAdapterV2, infer_task_type


def _rows(inputs: Any) -> NDArray[Any]:
    values = inputs.to_numpy() if hasattr(inputs, "to_numpy") else np.asarray(inputs)
    values = np.asarray(values)
    return np.asarray(values.reshape(1, -1) if values.ndim == 1 else values)


def _feature_names(model: Any, count: int) -> list[str]:
    names = getattr(model, "feature_names_in_", None)
    return [str(item) for item in names] if names is not None else [f"feature_{index}" for index in range(count)]


class SklearnModelAdapterV2(ModelAdapterV2):
    adapter_id = "sklearn_generic_v2"
    model_family = "sklearn_generic"

    def predict(self, inputs: Any) -> ModelPrediction:
        predictor = getattr(self.model, "predict", None)
        if not callable(predictor):
            raise TypeError(f"{self.adapter_id} requires model.predict")
        predictions = predictor(inputs)
        probabilities = self.model.predict_proba(inputs) if callable(getattr(self.model, "predict_proba", None)) else None
        decision_scores = self.model.decision_function(inputs) if callable(getattr(self.model, "decision_function", None)) else None
        if self.output_decoder:
            predictions = self.output_decoder(predictions)
        return ModelPrediction(
            predictions=_serializable(predictions),
            probabilities=_serializable(probabilities),
            model_type=type(self.model).__name__,
            adapter_id=self.adapter_id,
            metadata={
                "classes": _serializable(getattr(self.model, "classes_", None)),
                "decision_scores": _serializable(decision_scores),
                "task_type": self.task_type.value,
                "score_semantics": "probability" if probabilities is not None else "uncalibrated_decision_score" if decision_scores is not None else "prediction",
                "calibrated_probability": probabilities is not None,
            },
        )

    def capabilities(self) -> ModelCapabilities:
        base = super().capabilities()
        has_proba = callable(getattr(self.model, "predict_proba", None))
        has_decision = callable(getattr(self.model, "decision_function", None))
        descriptors = list(base.channels)
        if has_proba:
            descriptors.append(EvidenceChannelDescriptor("predict_proba", True, "native", "model.predict_proba"))
        if has_decision:
            descriptors.append(
                EvidenceChannelDescriptor(
                    "decision_function",
                    True,
                    "native",
                    "model.decision_function",
                    limitations=("Decision score is not a calibrated probability.",),
                )
            )
        return replace(
            base,
            predict_proba=has_proba,
            decision_function=has_decision,
            calibration=has_proba,
            feature_names=bool(self.feature_names()),
            class_names=hasattr(self.model, "classes_"),
            channels=tuple(descriptors),
        )


class SklearnLinearAdapter(SklearnModelAdapterV2):
    adapter_id = "sklearn_linear_v2"
    model_family = "sklearn_linear"

    def capabilities(self) -> ModelCapabilities:
        base = super().capabilities()
        return replace(
            base,
            local_contributions=True,
            global_importance=True,
            channels=(
                *base.channels,
                EvidenceChannelDescriptor("coefficients", True, "native", "model.coef_"),
                EvidenceChannelDescriptor("local_contributions", True, "derived_from_native", "x_i * coefficient_i"),
            ),
        )

    def extract_local_evidence(self, inputs: Any, prediction: ModelPrediction, context: ExplanationContext) -> LocalModelEvidence:
        values: NDArray[np.float64] = _rows(inputs)[0].astype(float)
        coefficients = np.asarray(self.model.coef_, dtype=float)
        if coefficients.ndim == 1:
            row = coefficients
        elif coefficients.shape[0] == 1:
            row = coefficients[0]
        else:
            predicted = prediction.predictions[0] if isinstance(prediction.predictions, list) else prediction.predictions
            classes = list(getattr(self.model, "classes_", range(coefficients.shape[0])))
            row = coefficients[classes.index(predicted)] if predicted in classes else coefficients[0]
        names = list(context.feature_names) or _feature_names(self.model, len(values))
        contributions = {name: float(value * weight) for name, value, weight in zip(names, values, row)}
        return LocalModelEvidence(
            channels={
                "contributions": contributions,
                "coefficients": {name: float(weight) for name, weight in zip(names, row)},
                "intercept": _serializable(getattr(self.model, "intercept_", None)),
                "contribution_method": "derived_native_linear_term_x_coefficient",
            },
            descriptors=(
                EvidenceChannelDescriptor("coefficients", True, "native", "model.coef_"),
                EvidenceChannelDescriptor("local_contributions", True, "derived_from_native", "x_i * coefficient_i"),
            ),
            limitations=("Linear terms describe model behavior and are not domain causality.",),
        )


class SklearnTreeAdapter(SklearnModelAdapterV2):
    adapter_id = "sklearn_tree_v2"
    model_family = "sklearn_tree"

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
                EvidenceChannelDescriptor("decision_path", True, "native", "model.tree_"),
                EvidenceChannelDescriptor("native_rules", True, "native", "model.tree_"),
                EvidenceChannelDescriptor("local_contributions", True, "derived_from_native", "tree node value transitions"),
            ),
        )

    def extract_local_evidence(self, inputs: Any, prediction: ModelPrediction, context: ExplanationContext) -> LocalModelEvidence:
        rows = _rows(inputs)
        row = rows[0]
        tree = self.model.tree_
        names = list(context.feature_names) or _feature_names(self.model, rows.shape[1])
        node = 0
        path: list[dict[str, Any]] = []
        contributions = {name: 0.0 for name in names}
        classes = list(getattr(self.model, "classes_", ()))
        predicted = prediction.predictions[0] if isinstance(prediction.predictions, list) else prediction.predictions
        class_index = classes.index(predicted) if predicted in classes else 0

        def node_value(node_id: int) -> float:
            values = np.asarray(tree.value[node_id], dtype=float).reshape(-1)
            if self.task_type == TaskType.REGRESSION:
                return float(values[0])
            total = float(values.sum())
            return float(values[class_index] / total) if total else 0.0

        while int(tree.children_left[node]) != int(tree.children_right[node]):
            feature_index = int(tree.feature[node])
            feature = names[feature_index]
            threshold = float(tree.threshold[node])
            direction = "left" if float(row[feature_index]) <= threshold else "right"
            child = int(tree.children_left[node] if direction == "left" else tree.children_right[node])
            delta = node_value(child) - node_value(node)
            contributions[feature] += delta
            path.append(
                {
                    "node_id": node,
                    "feature": feature,
                    "value": float(row[feature_index]),
                    "threshold": threshold,
                    "operator": "<=" if direction == "left" else ">",
                    "child": child,
                    "output_delta": delta,
                }
            )
            node = child
        return LocalModelEvidence(
            channels={
                "decision_path": path,
                "leaf_id": node,
                "leaf_samples": int(tree.n_node_samples[node]),
                "contributions": contributions,
                "contribution_method": "derived_native_tree_path_transition",
            },
            descriptors=(
                EvidenceChannelDescriptor("decision_path", True, "native", "model.tree_"),
                EvidenceChannelDescriptor("local_contributions", True, "derived_from_native", "tree node value transitions"),
            ),
            limitations=("Path contributions describe this fitted tree and are not causal effects.",),
        )


class SklearnEnsembleAdapter(SklearnModelAdapterV2):
    adapter_id = "sklearn_ensemble_v2"
    model_family = "sklearn_ensemble"

    def capabilities(self) -> ModelCapabilities:
        base = super().capabilities()
        tree_estimators = any(hasattr(item, "tree_") for item in np.asarray(getattr(self.model, "estimators_", ()), dtype=object).reshape(-1))
        return replace(
            base,
            uncertainty=True,
            global_importance=hasattr(self.model, "feature_importances_"),
            native_rules=tree_estimators,
            decision_path=tree_estimators,
            channels=(
                *base.channels,
                EvidenceChannelDescriptor("ensemble_votes", True, "native", "base estimator predictions"),
                EvidenceChannelDescriptor("ensemble_disagreement", True, "derived_from_native", "vote dispersion"),
                EvidenceChannelDescriptor("native_rules", tree_estimators, "native", "base estimator trees"),
            ),
        )

    def extract_local_evidence(self, inputs: Any, prediction: ModelPrediction, context: ExplanationContext) -> LocalModelEvidence:
        del prediction, context
        estimators = [item for item in np.asarray(getattr(self.model, "estimators_", ()), dtype=object).reshape(-1) if hasattr(item, "predict")]
        votes = [_serializable(item.predict(inputs)) for item in estimators]
        numeric = np.asarray([np.asarray(item).reshape(-1)[0] for item in votes], dtype=float) if votes else np.asarray([])
        disagreement = float(np.std(numeric)) if numeric.size else None
        importance = getattr(self.model, "feature_importances_", None)
        channels: dict[str, Any] = {
            "ensemble_votes": votes,
            "ensemble_disagreement": disagreement,
            "contribution_method": "native_ensemble_votes",
        }
        if importance is not None:
            names = self.feature_names() or [f"feature_{index}" for index in range(len(importance))]
            channels["global_importance"] = {name: float(value) for name, value in zip(names, importance)}
        return LocalModelEvidence(
            channels=channels,
            descriptors=(
                EvidenceChannelDescriptor("ensemble_votes", True, "native", "base estimator predictions"),
                EvidenceChannelDescriptor("ensemble_disagreement", True, "derived_from_native", "vote dispersion"),
            ),
            limitations=("Global importance is not substituted for a local contribution.",),
        )


class SklearnSVMAdapter(SklearnModelAdapterV2):
    adapter_id = "sklearn_svm_v2"
    model_family = "sklearn_svm"

    def capabilities(self) -> ModelCapabilities:
        base = super().capabilities()
        linear = str(getattr(self.model, "kernel", "linear")) == "linear" or hasattr(self.model, "coef_")
        surrogate = EvidenceChannelDescriptor(
            "local_contributions",
            True,
            "derived_from_native" if linear else "surrogate",
            "linear coefficient" if linear else "deterministic local linear approximation",
            fidelity_status="not_applicable" if linear else "measured",
            limitations=() if linear else ("Nonlinear-kernel contributions are surrogate, not native.",),
        )
        return replace(
            base,
            support_examples=hasattr(self.model, "support_vectors_"),
            local_contributions=True,
            channels=(*base.channels, EvidenceChannelDescriptor("support_vectors", hasattr(self.model, "support_vectors_"), "native", "model.support_vectors_"), surrogate),
        )

    def extract_local_evidence(self, inputs: Any, prediction: ModelPrediction, context: ExplanationContext) -> LocalModelEvidence:
        values: NDArray[np.float64] = _rows(inputs)[0].astype(float)
        names = list(context.feature_names) or _feature_names(self.model, len(values))
        if hasattr(self.model, "coef_"):
            coefficient = np.asarray(self.model.coef_, dtype=float).reshape(-1, len(values))[0]
            contributions = {name: float(value * weight) for name, value, weight in zip(names, values, coefficient)}
            origin: EvidenceOrigin = "derived_from_native"
            fidelity = None
            method = "derived_native_linear_svm_term"
        else:
            scale = np.maximum(np.abs(values) * 0.01, 1e-3)
            perturbations = [values]
            for index in range(len(values)):
                low, high = values.copy(), values.copy()
                low[index] -= scale[index]
                high[index] += scale[index]
                perturbations.extend((low, high))
            matrix = np.asarray(perturbations)
            scorer = getattr(self.model, "decision_function", None)
            if not callable(scorer):
                scorer = self.model.predict
            target = np.asarray(scorer(matrix), dtype=float)
            target = target[:, 0] if target.ndim > 1 else target
            design = np.column_stack([np.ones(len(matrix)), matrix - values])
            coefficients, *_ = np.linalg.lstsq(design, target, rcond=None)
            fitted = design @ coefficients
            denominator = float(np.sum((target - target.mean()) ** 2))
            fidelity = 1.0 if denominator <= 1e-15 else max(0.0, 1.0 - float(np.sum((target - fitted) ** 2)) / denominator)
            contributions = {name: float(weight * value) for name, value, weight in zip(names, values, coefficients[1:])}
            origin = "surrogate"
            method = "surrogate_local_linear_svm"
        return LocalModelEvidence(
            channels={
                "contributions": contributions,
                "contribution_method": method,
                "surrogate_fidelity": fidelity,
                "nearest_support_vectors": _serializable(getattr(self.model, "support_vectors_", ())[:3]),
            },
            descriptors=(
                EvidenceChannelDescriptor(
                    "local_contributions",
                    True,
                    origin,
                    method,
                    fidelity_status="measured" if fidelity is not None else "not_applicable",
                    fidelity=fidelity,
                    limitations=() if origin != "surrogate" else ("Nonlinear-kernel contributions are surrogate.",),
                ),
            ),
            limitations=() if origin != "surrogate" else ("Surrogate reasons require local fidelity >= 0.90 to enter top reasons.",),
        )


class SklearnKNNAdapter(SklearnModelAdapterV2):
    adapter_id = "sklearn_knn_v2"
    model_family = "sklearn_knn"

    def capabilities(self) -> ModelCapabilities:
        base = super().capabilities()
        return replace(
            base,
            nearest_neighbors=True,
            support_examples=True,
            channels=(*base.channels, EvidenceChannelDescriptor("nearest_neighbors", True, "native", "model.kneighbors")),
        )

    def extract_local_evidence(self, inputs: Any, prediction: ModelPrediction, context: ExplanationContext) -> LocalModelEvidence:
        del prediction, context
        count = min(int(getattr(self.model, "n_neighbors", 5)), len(getattr(self.model, "_fit_X", ())))
        distances, indices = self.model.kneighbors(inputs, n_neighbors=count)
        labels = getattr(self.model, "_y", None)
        neighbor_labels = [labels[index] for index in indices[0]] if labels is not None else []
        return LocalModelEvidence(
            channels={
                "neighbor_indices": _serializable(indices[0]),
                "neighbor_distances": _serializable(distances[0]),
                "neighbor_labels": _serializable(neighbor_labels),
                "local_density": None if not len(distances[0]) else float(1.0 / (1.0 + np.mean(distances[0]))),
                "contribution_method": "native_nearest_neighbors",
            },
            descriptors=(EvidenceChannelDescriptor("nearest_neighbors", True, "native", "model.kneighbors"),),
            limitations=("Neighbor similarity supports comparison but is not a direct feature contribution.",),
        )


class SklearnNaiveBayesAdapter(SklearnModelAdapterV2):
    adapter_id = "sklearn_naive_bayes_v2"
    model_family = "sklearn_naive_bayes"

    def capabilities(self) -> ModelCapabilities:
        base = super().capabilities()
        return replace(
            base,
            local_contributions=hasattr(self.model, "theta_") and hasattr(self.model, "var_"),
            channels=(
                *base.channels,
                EvidenceChannelDescriptor("class_priors", hasattr(self.model, "class_prior_"), "native", "model.class_prior_"),
                EvidenceChannelDescriptor("log_likelihood_terms", hasattr(self.model, "theta_"), "derived_from_native", "Gaussian log likelihood"),
            ),
        )

    def extract_local_evidence(self, inputs: Any, prediction: ModelPrediction, context: ExplanationContext) -> LocalModelEvidence:
        values: NDArray[np.float64] = _rows(inputs)[0].astype(float)
        names = list(context.feature_names) or _feature_names(self.model, len(values))
        classes = list(getattr(self.model, "classes_", ()))
        predicted = prediction.predictions[0] if isinstance(prediction.predictions, list) else prediction.predictions
        class_index = classes.index(predicted) if predicted in classes else 0
        means = np.asarray(self.model.theta_[class_index], dtype=float)
        variances = np.maximum(np.asarray(self.model.var_[class_index], dtype=float), 1e-15)
        terms = -0.5 * (np.log(2.0 * np.pi * variances) + ((values - means) ** 2) / variances)
        return LocalModelEvidence(
            channels={
                "contributions": {name: float(value) for name, value in zip(names, terms)},
                "class_prior": float(np.asarray(self.model.class_prior_)[class_index]),
                "contribution_method": "derived_native_gaussian_log_likelihood",
            },
            descriptors=(EvidenceChannelDescriptor("log_likelihood_terms", True, "derived_from_native", "Gaussian log likelihood"),),
            limitations=("Naive Bayes explanation inherits the conditional-independence assumption.",),
        )


class SklearnRegressorAdapter(SklearnModelAdapterV2):
    adapter_id = "sklearn_regressor_v2"
    model_family = "sklearn_regressor"

    def __init__(self, model: Any, *, task: str | TaskType = TaskType.REGRESSION, output_decoder: Any = None):
        super().__init__(model, task=task, output_decoder=output_decoder)


class SklearnPipelineAdapter(SklearnModelAdapterV2):
    adapter_id = "sklearn_pipeline_v2"
    model_family = "sklearn_pipeline"

    def __init__(self, model: Any, *, task: str | TaskType = "auto", output_decoder: Any = None):
        super().__init__(model, task=task, output_decoder=output_decoder)
        if not hasattr(model, "steps"):
            raise TypeError("SklearnPipelineAdapter requires sklearn Pipeline-compatible steps")

    def input_schema(self) -> ModelInputSchema:
        original = tuple(self.feature_names())
        transformed: tuple[str, ...] = ()
        provenance: dict[str, tuple[str, ...]] = {}
        preprocessor = self.model[:-1]
        getter = getattr(preprocessor, "get_feature_names_out", None)
        if callable(getter):
            try:
                transformed = tuple(str(item) for item in getter(original or None))
            except (TypeError, ValueError):
                transformed = tuple(str(item) for item in getter())
            for name in transformed:
                matches = tuple(item for item in original if name == item or name.endswith(f"__{item}") or f"_{item}_" in name)
                provenance[name] = matches or (name,)
        return ModelInputSchema(
            feature_names=original,
            transformed_feature_names=transformed,
            feature_provenance=provenance,
        )

    def capabilities(self) -> ModelCapabilities:
        final = self.model.steps[-1][1]
        delegated = resolve_sklearn_adapter(final, task=self.task_type).capabilities()
        return replace(delegated, feature_names=True)

    def extract_local_evidence(self, inputs: Any, prediction: ModelPrediction, context: ExplanationContext) -> LocalModelEvidence:
        transformed = self.model[:-1].transform(inputs)
        final = self.model.steps[-1][1]
        schema = self.input_schema()
        names = schema.transformed_feature_names or tuple(f"feature_{index}" for index in range(_rows(transformed).shape[1]))
        delegated = resolve_sklearn_adapter(final, task=self.task_type)
        local = delegated.extract_local_evidence(
            transformed,
            delegated.predict(transformed),
            replace(context, feature_names=names),
        )
        channels = dict(local.channels)
        channels["pipeline_steps"] = [name for name, _ in self.model.steps]
        channels["feature_provenance"] = {key: list(value) for key, value in schema.feature_provenance.items()}
        return LocalModelEvidence(
            channels=channels,
            descriptors=local.descriptors,
            missing_channels=local.missing_channels,
            limitations=local.limitations,
        )


def sklearn_family(model: Any) -> str:
    name = type(model).__name__
    if hasattr(model, "steps"):
        return "pipeline"
    if name in {"KNeighborsClassifier", "KNeighborsRegressor"} or callable(getattr(model, "kneighbors", None)):
        return "knn"
    if name.endswith("NB") or hasattr(model, "class_log_prior_"):
        return "naive_bayes"
    if name in {"SVC", "SVR", "NuSVC", "NuSVR", "OneClassSVM"} or "SVC" in name or "SVR" in name:
        return "svm"
    if hasattr(model, "tree_"):
        return "tree"
    if hasattr(model, "estimators_"):
        return "ensemble"
    if hasattr(model, "coef_"):
        return "linear"
    if infer_task_type(model) == TaskType.REGRESSION:
        return "regressor"
    return "generic"


def resolve_sklearn_adapter(model: Any, *, task: str | TaskType = "auto", output_decoder: Any = None) -> ModelAdapterV2:
    family = sklearn_family(model)
    adapters = {
        "pipeline": SklearnPipelineAdapter,
        "knn": SklearnKNNAdapter,
        "naive_bayes": SklearnNaiveBayesAdapter,
        "svm": SklearnSVMAdapter,
        "tree": SklearnTreeAdapter,
        "ensemble": SklearnEnsembleAdapter,
        "linear": SklearnLinearAdapter,
        "regressor": SklearnRegressorAdapter,
        "generic": SklearnModelAdapterV2,
    }
    return adapters[family](model, task=task, output_decoder=output_decoder)
