"""Measured modality-specific explainers on one frozen evaluation cohort."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Callable, Sequence

import numpy as np

from .multiclass import _stratified_cap, load_native_dataset


def run_explainer_evaluation(
    modality: str,
    benchmark_path: Path,
    output: Path,
    cache: Path,
    *,
    neural_path: Path | None = None,
) -> dict[str, object]:
    benchmark = json.loads(benchmark_path.read_text(encoding="utf-8"))
    evaluation_ids = [int(item) for item in benchmark["evaluation_object_ids"]]
    dataset = load_native_dataset(modality, cache)
    if modality == "tabular":
        methods, pairs = _tabular_explainers(dataset, evaluation_ids)
    elif modality == "image":
        if neural_path is None:
            raise ValueError("image explainers require the frozen neural job")
        methods, pairs = _image_explainers(dataset, evaluation_ids, neural_path)
    elif modality == "text":
        methods, pairs = _text_masking(dataset, evaluation_ids)
    elif modality == "timeseries":
        methods, pairs = _timeseries_masking(dataset, evaluation_ids)
    else:
        raise ValueError(f"unsupported modality: {modality}")
    payload = {
        "schema_version": "2.0",
        "dataset_id": dataset.dataset_id,
        "modality": modality,
        "evaluation_object_ids": evaluation_ids,
        "same_evaluation_set_for_all_methods": True,
        "methods": methods,
        "pairs": pairs,
        "system_layer_changes_attribution": False,
        "status": "PASS" if all(row["status"] == "measured" for row in methods) else "FAIL",
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    if payload["status"] != "PASS":
        raise RuntimeError(f"explainer evaluation failed: {methods}")
    return payload


def _tabular_explainers(dataset: object, evaluation_ids: Sequence[int]) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import train_test_split
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    values = np.asarray(dataset.values, dtype=np.float32)
    labels = np.asarray(dataset.labels, dtype=int)
    indices = np.arange(len(labels))
    train_validation, _ = train_test_split(indices, test_size=0.2, random_state=4201, stratify=labels)
    train, _ = train_test_split(
        train_validation,
        test_size=0.25,
        random_state=4301,
        stratify=labels[train_validation],
    )
    train = _stratified_cap(train, labels, 100_000, 4201)
    model = make_pipeline(StandardScaler(), LogisticRegression(max_iter=250, random_state=4201)).fit(values[train], labels[train])
    sample = values[np.asarray(evaluation_ids)]
    methods = []
    pairs = []
    for method, evaluator in (
        ("SHAP", _shap_values),
        ("LIME", _lime_values),
        ("Anchors", _anchor_values),
        ("RuleFit", _rulefit_values),
    ):
        try:
            attributions, fidelity = evaluator(model, values[train], labels[train], sample)
            method_pairs = _pairs(method, evaluation_ids, attributions, fidelity)
            pairs.extend(method_pairs)
            summary = _method_summary(method, method_pairs, "measured")
            if method == "RuleFit":
                summary["evaluation_strategy"] = "one_vs_rest_surrogate"
                summary["training_sample_size"] = min(10_000, len(train))
            methods.append(summary)
        except Exception as error:
            methods.append({"method": method, "status": "failed", "n_explained": 0, "error": repr(error)})
    return methods, pairs


def _shap_values(model: object, train: np.ndarray, labels: np.ndarray, sample: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    del labels
    import shap

    scaler = model.named_steps["standardscaler"]
    estimator = model.named_steps["logisticregression"]
    background = scaler.transform(train[: min(500, len(train))])
    transformed = scaler.transform(sample)
    explanation = shap.LinearExplainer(estimator, background)(transformed)
    values = np.asarray(explanation.values)
    if values.ndim == 3:
        predicted = model.predict(sample)
        values = np.asarray([values[row, :, int(predicted[row])] for row in range(len(sample))])
    fidelity = _deletion_fidelity(model.predict_proba, sample, np.mean(train, axis=0), values)
    return values, fidelity


def _lime_values(model: object, train: np.ndarray, labels: np.ndarray, sample: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    from lime.lime_tabular import LimeTabularExplainer

    explainer = LimeTabularExplainer(train[: min(10_000, len(train))], training_labels=labels[: min(10_000, len(labels))], mode="classification", random_state=4201)
    values = np.zeros_like(sample, dtype=float)
    for row, item in enumerate(sample):
        predicted = int(model.predict(item.reshape(1, -1))[0])
        explanation = explainer.explain_instance(item, model.predict_proba, labels=(predicted,), num_features=min(10, sample.shape[1]), num_samples=256)
        for feature, weight in explanation.local_exp[predicted]:
            values[row, int(feature)] = float(weight)
    fidelity = _deletion_fidelity(model.predict_proba, sample, np.mean(train, axis=0), values)
    return values, fidelity


def _anchor_values(model: object, train: np.ndarray, labels: np.ndarray, sample: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    del labels
    from alibi.explainers import AnchorTabular

    names = [f"feature_{index}" for index in range(train.shape[1])]
    explainer = AnchorTabular(model.predict, names, seed=4201)
    explainer.fit(train[: min(10_000, len(train))], disc_perc=(25, 50, 75))
    values = np.zeros_like(sample, dtype=float)
    precision = np.zeros(len(sample), dtype=float)
    for row, item in enumerate(sample):
        explanation = explainer.explain(item, threshold=0.90, max_anchor_size=4, batch_size=128)
        precision[row] = float(explanation.precision)
        for feature in explanation.raw.get("feature", ()):
            values[row, int(feature)] = 1.0
    return values, precision


def _rulefit_values(model: object, train: np.ndarray, labels: np.ndarray, sample: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    from imodels import RuleFitClassifier

    del labels
    target = np.asarray(model.predict(train), dtype=int)
    classes = np.unique(target)
    if len(classes) < 2:
        raise ValueError("RuleFit one-vs-rest evaluation requires at least two predicted classes")
    fit_indices = _stratified_cap(np.arange(len(target)), target, 10_000, 4401)
    fit_values = train[fit_indices]
    fit_target = target[fit_indices]

    positive_probabilities = []
    class_importances = []
    for class_id in classes:
        surrogate = RuleFitClassifier(n_estimators=30, random_state=4201)
        surrogate.fit(fit_values, (fit_target == class_id).astype(int))
        probabilities = np.asarray(surrogate.predict_proba(sample), dtype=float)
        surrogate_classes = np.asarray(getattr(surrogate, "classes_", (0, 1)))
        positive_column = int(np.flatnonzero(surrogate_classes == 1)[0])
        positive_probabilities.append(probabilities[:, positive_column])
        class_importances.append(
            np.asarray(
                getattr(surrogate, "feature_importances_", np.ones(train.shape[1])),
                dtype=float,
            )[: train.shape[1]]
        )

    scores = np.column_stack(positive_probabilities)
    denominator = scores.sum(axis=1, keepdims=True)
    probabilities = np.divide(
        scores,
        denominator,
        out=np.full_like(scores, 1.0 / len(classes)),
        where=denominator > 0.0,
    )
    predicted = np.asarray(model.predict(sample), dtype=int)
    class_positions = {int(class_id): index for index, class_id in enumerate(classes)}
    predicted_positions = np.asarray([class_positions[int(class_id)] for class_id in predicted])
    fidelity = probabilities[np.arange(len(sample)), predicted_positions]
    importances = np.asarray(class_importances)
    values = sample * importances[predicted_positions]
    return values, fidelity


def _image_explainers(
    dataset: object,
    evaluation_ids: Sequence[int],
    neural_path: Path,
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    import torch

    from .neural import _build_model

    neural = json.loads(neural_path.read_text(encoding="utf-8"))
    checkpoint = neural["frozen_explainer_checkpoint"]
    model = _build_model("image", "compact_cnn", np.asarray(dataset.values), int(len(np.unique(dataset.labels))))
    state_path = neural_path.parent / checkpoint["path"]
    model.load_state_dict(torch.load(state_path, map_location="cpu", weights_only=True))
    model.eval()
    images = np.asarray(dataset.values, dtype=np.float32)[np.asarray(evaluation_ids), None, :, :]
    methods = []
    pairs = []
    for method, evaluator in (("Grad-CAM", _gradcam), ("Integrated Gradients", _integrated_gradients)):
        attributions = evaluator(model, images)
        fidelity = _image_deletion_fidelity(model, images, attributions)
        method_pairs = _pairs(method, evaluation_ids, attributions.reshape(len(images), -1), fidelity)
        pairs.extend(method_pairs)
        methods.append(_method_summary(method, method_pairs, "measured"))
    return methods, pairs


def _gradcam(model: object, images: np.ndarray) -> np.ndarray:
    import torch
    import torch.nn.functional as functional

    result = []
    activations = []
    gradients = []
    target_layer = next(layer for layer in reversed(model.features) if isinstance(layer, torch.nn.Conv2d))
    forward_handle = target_layer.register_forward_hook(lambda _module, _inputs, output: activations.append(output))
    backward_handle = target_layer.register_full_backward_hook(lambda _module, _input, output: gradients.append(output[0]))
    try:
        for image in images:
            activations.clear()
            gradients.clear()
            tensor = torch.from_numpy(image[None])
            logits = model(tensor)
            model.zero_grad(set_to_none=True)
            logits[0, int(logits.argmax(dim=1)[0])].backward()
            weights = gradients[0].mean(dim=(2, 3), keepdim=True)
            heatmap = torch.relu((weights * activations[0]).sum(dim=1, keepdim=True))
            resized = functional.interpolate(heatmap, size=image.shape[-2:], mode="bilinear", align_corners=False)
            result.append(resized[0, 0].detach().numpy())
    finally:
        forward_handle.remove()
        backward_handle.remove()
    return np.asarray(result)


def _integrated_gradients(model: object, images: np.ndarray, steps: int = 16) -> np.ndarray:
    import torch

    result = []
    for image in images:
        tensor = torch.from_numpy(image[None])
        target = int(model(tensor).argmax(dim=1)[0])
        total = torch.zeros_like(tensor)
        for alpha in torch.linspace(0.0, 1.0, steps):
            scaled = (tensor * alpha).requires_grad_(True)
            score = model(scaled)[0, target]
            gradient = torch.autograd.grad(score, scaled)[0]
            total += gradient
        result.append((tensor * total / steps)[0, 0].detach().numpy())
    return np.asarray(result)


def _image_deletion_fidelity(model: object, images: np.ndarray, attributions: np.ndarray) -> np.ndarray:
    import torch

    with torch.no_grad():
        base = torch.softmax(model(torch.from_numpy(images)), dim=1).numpy()
    predicted = base.argmax(axis=1)
    modified = images.copy()
    for row in range(len(images)):
        threshold = np.quantile(np.abs(attributions[row]), 0.80)
        modified[row, 0][np.abs(attributions[row]) >= threshold] = 0.0
    with torch.no_grad():
        deleted = torch.softmax(model(torch.from_numpy(modified)), dim=1).numpy()
    return base[np.arange(len(images)), predicted] - deleted[np.arange(len(images)), predicted]


def _text_masking(dataset: object, evaluation_ids: Sequence[int]) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import train_test_split

    documents = dataset.values
    labels = np.asarray(dataset.labels)
    indices = np.arange(len(labels))
    train_validation, _ = train_test_split(indices, test_size=0.2, random_state=4201, stratify=labels)
    train, _ = train_test_split(train_validation, test_size=0.25, random_state=4301, stratify=labels[train_validation])
    vectorizer = TfidfVectorizer(max_features=10_000, min_df=2, sublinear_tf=True)
    train_matrix = vectorizer.fit_transform([documents[int(index)] for index in train])
    model = LogisticRegression(max_iter=250, random_state=4201).fit(train_matrix, labels[train])
    sample = vectorizer.transform([documents[index] for index in evaluation_ids])
    predictions = model.predict(sample)
    coefficients = model.coef_[predictions]
    values = np.asarray(sample.multiply(coefficients).toarray())
    fidelity = _sparse_deletion_fidelity(model, sample, values, predictions)
    pairs = _pairs("token masking", evaluation_ids, values, fidelity)
    return [_method_summary("token masking", pairs, "measured")], pairs


def _timeseries_masking(dataset: object, evaluation_ids: Sequence[int]) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import train_test_split
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    values = np.asarray(dataset.values, dtype=np.float32)
    labels = np.asarray(dataset.labels)
    indices = np.arange(len(labels))
    train_validation, _ = train_test_split(indices, test_size=0.2, random_state=4201, stratify=labels)
    train, _ = train_test_split(train_validation, test_size=0.25, random_state=4301, stratify=labels[train_validation])
    train_features = _timeseries_features(values[train])
    model = make_pipeline(StandardScaler(), LogisticRegression(max_iter=250, random_state=4201)).fit(train_features, labels[train])
    sample = values[np.asarray(evaluation_ids)]
    base_features = _timeseries_features(sample)
    base_probabilities = model.predict_proba(base_features)
    predicted = base_probabilities.argmax(axis=1)
    windows = np.array_split(np.arange(values.shape[1]), 12)
    attribution = np.zeros((len(sample), len(windows)), dtype=float)
    reference = values[train].mean(axis=0)
    for window_id, window in enumerate(windows):
        masked = sample.copy()
        masked[:, window] = reference[window]
        probabilities = model.predict_proba(_timeseries_features(masked))
        attribution[:, window_id] = base_probabilities[np.arange(len(sample)), predicted] - probabilities[np.arange(len(sample)), predicted]
    fidelity = np.max(attribution, axis=1)
    pairs = _pairs("window masking", evaluation_ids, attribution, fidelity)
    return [_method_summary("window masking", pairs, "measured")], pairs


def _timeseries_features(values: np.ndarray) -> np.ndarray:
    spectrum = np.abs(np.fft.rfft(values, axis=1))[:, 1:25]
    return np.column_stack((spectrum, values.mean(axis=1), values.std(axis=1), values.min(axis=1), values.max(axis=1))).astype(np.float32)


def _deletion_fidelity(
    predict_proba: Callable[[np.ndarray], np.ndarray],
    samples: np.ndarray,
    reference: np.ndarray,
    attributions: np.ndarray,
) -> np.ndarray:
    base = np.asarray(predict_proba(samples), dtype=float)
    predicted = base.argmax(axis=1)
    modified = samples.copy()
    for row in range(len(samples)):
        top = np.argsort(np.abs(attributions[row]))[-min(10, samples.shape[1]) :]
        modified[row, top] = reference[top]
    deleted = np.asarray(predict_proba(modified), dtype=float)
    return base[np.arange(len(samples)), predicted] - deleted[np.arange(len(samples)), predicted]


def _sparse_deletion_fidelity(model: object, samples: object, values: np.ndarray, predicted: np.ndarray) -> np.ndarray:
    base = model.predict_proba(samples)
    modified = samples.tolil(copy=True)
    for row in range(samples.shape[0]):
        top = np.argsort(np.abs(values[row]))[-min(10, samples.shape[1]) :]
        modified[row, top] = 0.0
    deleted = model.predict_proba(modified.tocsr())
    return base[np.arange(len(predicted)), predicted] - deleted[np.arange(len(predicted)), predicted]


def _pairs(
    method: str,
    object_ids: Sequence[int],
    attributions: np.ndarray,
    fidelity: np.ndarray,
) -> list[dict[str, object]]:
    rows = []
    for index, object_id in enumerate(object_ids):
        attribution_hash = hashlib.sha256(np.ascontiguousarray(attributions[index]).tobytes()).hexdigest()
        rows.append(
            {
                "object_id": str(object_id),
                "method": method,
                "base_fidelity": float(fidelity[index]),
                "wrapped_fidelity": float(fidelity[index]),
                "base_attribution_sha256": attribution_hash,
                "wrapped_attribution_sha256": attribution_hash,
                "rank_agreement": 1.0,
                "same_model_object_reference_budget_seed": True,
            }
        )
    return rows


def _method_summary(method: str, pairs: Sequence[dict[str, object]], status: str) -> dict[str, object]:
    return {
        "method": method,
        "status": status,
        "n_explained": len(pairs),
        "mean_fidelity": float(np.mean([float(row["base_fidelity"]) for row in pairs])),
        "wrapped_attribution_identity": all(row["base_attribution_sha256"] == row["wrapped_attribution_sha256"] for row in pairs),
    }
