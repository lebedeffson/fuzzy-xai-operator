"""Native multiclass benchmark used by the final Q1 heavy CI matrix."""

from __future__ import annotations

import hashlib
import json
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from time import perf_counter
from typing import Mapping, Sequence

import numpy as np


@dataclass(frozen=True)
class NativeDataset:
    dataset_id: str
    modality: str
    values: np.ndarray | tuple[str, ...]
    labels: np.ndarray
    source: str
    license: str
    raw_sha256: str


def load_native_dataset(modality: str, cache: Path) -> NativeDataset:
    from fuzzyxai.q1_validation import real_benchmarks as legacy

    cache.mkdir(parents=True, exist_ok=True)
    if modality == "tabular":
        from sklearn.datasets import fetch_covtype

        bunch = fetch_covtype(data_home=cache, download_if_missing=True)
        values = np.asarray(bunch.data, dtype=np.float32)
        labels = np.asarray(bunch.target, dtype=int) - 1
        return NativeDataset(
            "uci_covertype",
            modality,
            values,
            labels,
            "https://archive.ics.uci.edu/dataset/180/cover+type",
            "CC BY 4.0",
            _hash(values, labels),
        )
    if modality == "image":
        base = "https://raw.githubusercontent.com/zalandoresearch/fashion-mnist/master/data/fashion"
        names = (
            "train-images-idx3-ubyte.gz",
            "train-labels-idx1-ubyte.gz",
            "t10k-images-idx3-ubyte.gz",
            "t10k-labels-idx1-ubyte.gz",
        )
        paths = [legacy._download(f"{base}/{name}", cache / name) for name in names]
        values = np.concatenate((legacy._read_idx_images(paths[0]), legacy._read_idx_images(paths[2]))).astype(np.float32) / 255.0
        labels = np.concatenate((legacy._read_idx_labels(paths[1]), legacy._read_idx_labels(paths[3]))).astype(int)
        return NativeDataset("fashion_mnist", modality, values, labels, "https://github.com/zalandoresearch/fashion-mnist", "MIT", legacy._files_sha(paths))
    if modality == "text":
        from sklearn.datasets import fetch_20newsgroups

        bunch = fetch_20newsgroups(data_home=cache, subset="all", remove=("headers", "footers", "quotes"), shuffle=False)
        documents = tuple(str(item) for item in bunch.data)
        labels = np.asarray(bunch.target, dtype=int)
        raw = "\n\0\n".join(documents).encode("utf-8", errors="replace")
        return NativeDataset("20newsgroups", modality, documents, labels, "https://scikit-learn.org/stable/modules/generated/sklearn.datasets.fetch_20newsgroups.html", "source-specific post rights", hashlib.sha256(raw + labels.tobytes()).hexdigest())
    if modality == "timeseries":
        base = "https://zenodo.org/records/11190880/files"
        train_path = legacy._download(f"{base}/ElectricDevices_TRAIN.ts?download=1", cache / "ElectricDevices_TRAIN.ts")
        test_path = legacy._download(f"{base}/ElectricDevices_TEST.ts?download=1", cache / "ElectricDevices_TEST.ts")
        train_values, train_labels = legacy._read_ts(train_path)
        test_values, test_labels = legacy._read_ts(test_path)
        values = np.concatenate((train_values, test_values)).astype(np.float32)
        original = np.concatenate((train_labels, test_labels))
        classes = {value: index for index, value in enumerate(sorted(set(float(item) for item in original)))}
        labels = np.asarray([classes[float(value)] for value in original], dtype=int)
        return NativeDataset("ucr_electric_devices", modality, values, labels, "https://zenodo.org/records/11190880", "CC BY 4.0", legacy._files_sha((train_path, test_path)))
    raise ValueError(f"unsupported modality: {modality}")


def run_native_multiclass(
    modality: str,
    output: Path,
    cache: Path,
    *,
    seeds: Sequence[int] = (4201, 4202, 4203, 4204, 4205),
) -> dict[str, object]:
    from sklearn.calibration import CalibratedClassifierCV
    from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import (
        accuracy_score,
        balanced_accuracy_score,
        confusion_matrix,
        f1_score,
        precision_recall_fscore_support,
    )
    from sklearn.model_selection import train_test_split
    from sklearn.neural_network import MLPClassifier
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    from sklearn.svm import LinearSVC

    dataset = load_native_dataset(modality, cache)
    all_indices = np.arange(len(dataset.labels))
    runs: list[dict[str, object]] = []
    split_records: list[dict[str, object]] = []
    object_predictions: list[dict[str, object]] = []
    evaluation_ids: list[int] = []
    explanation_pairs: list[dict[str, object]] = []
    for seed_index, seed in enumerate(seeds):
        train_val, test = train_test_split(all_indices, test_size=0.2, random_state=seed, stratify=dataset.labels)
        train, validation = train_test_split(train_val, test_size=0.25, random_state=seed + 100, stratify=dataset.labels[train_val])
        split_records.append(
            {
                "seed": seed,
                "train_sha256": _hash(train),
                "validation_sha256": _hash(validation),
                "test_sha256": _hash(test),
                "test_used_for_selection": False,
            }
        )
        train_cap = _stratified_cap(train, dataset.labels, 100_000, seed)
        validation_cap = _stratified_cap(validation, dataset.labels, 10_000, seed + 2)
        test_cap = _stratified_cap(test, dataset.labels, 10_000, seed + 1)
        x_train, x_validation, x_test = _features(dataset, train_cap, validation_cap, test_cap)
        y_train = dataset.labels[train_cap]
        y_validation = dataset.labels[validation_cap]
        y_test = dataset.labels[test_cap]
        models: list[tuple[str, str, object]] = [
            ("logistic_regression", "linear", make_pipeline(StandardScaler(), LogisticRegression(max_iter=250, random_state=seed))),
            ("random_forest", "tree", RandomForestClassifier(n_estimators=50, max_depth=14, n_jobs=1, random_state=seed)),
            ("hist_gradient_boosting", "boosting", HistGradientBoostingClassifier(max_iter=80, random_state=seed)),
            ("mlp", "neural_mlp", make_pipeline(StandardScaler(), MLPClassifier(hidden_layer_sizes=(32,), max_iter=35, early_stopping=True, random_state=seed))),
        ]
        if modality == "text":
            models = [
                ("tfidf_logistic", "linear", LogisticRegression(max_iter=250, random_state=seed)),
                ("calibrated_linear_svc", "linear_svc", CalibratedClassifierCV(LinearSVC(random_state=seed), cv=3)),
            ]
        for model_id, family, model in models:
            started = perf_counter()
            model.fit(x_train, y_train)
            fit_seconds = perf_counter() - started
            validation_probabilities = np.asarray(model.predict_proba(x_validation), dtype=float)
            probabilities = np.asarray(model.predict_proba(x_test), dtype=float)
            predictions = probabilities.argmax(axis=1)
            validation_predictions = validation_probabilities.argmax(axis=1)
            confidence_threshold = float(np.quantile(validation_probabilities.max(axis=1), 0.25))
            class_counts = Counter(int(item) for item in y_train)
            rare_cutoff = float(np.quantile(list(class_counts.values()), 0.25))
            rare_classes = sorted(class_id for class_id, size in class_counts.items() if size <= rare_cutoff)
            precision, recall, per_f1, support = precision_recall_fscore_support(y_test, predictions, labels=np.arange(len(np.unique(dataset.labels))), zero_division=0)
            row = {
                "seed": seed,
                "model_id": model_id,
                "family": family,
                "n_train_available": len(train),
                "n_train_used": len(train_cap),
                "n_validation_available": len(validation),
                "n_validation_used": len(validation_cap),
                "n_test_available": len(test),
                "n_test_used": len(test_cap),
                "fit_seconds": fit_seconds,
                "accuracy": float(accuracy_score(y_test, predictions)),
                "balanced_accuracy": float(balanced_accuracy_score(y_test, predictions)),
                "macro_f1": float(f1_score(y_test, predictions, average="macro")),
                "micro_f1": float(f1_score(y_test, predictions, average="micro")),
                "weighted_f1": float(f1_score(y_test, predictions, average="weighted")),
                "ece": _ece(probabilities, y_test),
                "validation_ece": _ece(validation_probabilities, y_validation),
                "validation_accuracy": float(accuracy_score(y_validation, validation_predictions)),
                "selection_thresholds": {
                    "low_confidence": confidence_threshold,
                    "rare_class_max_train_count": rare_cutoff,
                },
                "rare_classes": rare_classes,
                "confusion_matrix": confusion_matrix(y_test, predictions).tolist(),
                "per_class": [
                    {"class_id": index, "precision": float(precision[index]), "recall": float(recall[index]), "f1": float(per_f1[index]), "support": int(support[index])}
                    for index in range(len(support))
                ],
                "subgroups": _subgroup_metrics(y_test, predictions, rare_classes),
                "error_taxonomy": _error_taxonomy(y_test, predictions, probabilities, rare_classes),
                "status": "measured",
            }
            runs.append(row)
            if model_id == models[0][0]:
                for position, object_index in enumerate(test_cap):
                    object_predictions.append(
                        {
                            "seed": seed,
                            "object_id": int(object_index),
                            "true_class": int(y_test[position]),
                            "predicted_class": int(predictions[position]),
                            "confidence": float(probabilities[position].max()),
                            "correct": bool(predictions[position] == y_test[position]),
                            "class_probability": float(probabilities[position, y_test[position]]),
                            "rare_class": bool(int(y_test[position]) in rare_classes),
                            "low_confidence": bool(float(probabilities[position].max()) <= confidence_threshold),
                        }
                    )
                if seed_index == 0:
                    required = 1000 if modality == "tabular" else 500
                    evaluation_ids = _evaluation_sample(
                        test_cap,
                        y_test,
                        predictions,
                        probabilities.max(axis=1),
                        required,
                        seed,
                    )
                    explanation_pairs = _evaluate_explanation_pairs(
                        model,
                        x_train,
                        x_test,
                        test_cap,
                        evaluation_ids,
                        modality,
                    )
    payload = {
        "schema_version": "2.0",
        "dataset": {
            "dataset_id": dataset.dataset_id,
            "modality": modality,
            "native_class_count": int(len(np.unique(dataset.labels))),
            "n_objects": len(dataset.labels),
            "source": dataset.source,
            "license": dataset.license,
            "raw_sha256": dataset.raw_sha256,
        },
        "seeds": list(seeds),
        "splits": split_records,
        "models": runs,
        "object_predictions": object_predictions,
        "evaluation_object_ids": evaluation_ids,
        "evaluation_sample_frozen_before_explainer_comparison": True,
        "evaluation_sampling": "stratified by native class, correctness and validation-defined confidence band",
        "explanation_evaluation": {
            "pairs": explanation_pairs,
            "measured_methods": sorted({str(row["method"]) for row in explanation_pairs}),
            "required_method_coverage": _required_method_coverage(modality, explanation_pairs),
            "system_layer_changes_attribution": False,
        },
        "status": "PASS",
        "limitations": ["training caps are disclosed per run", "benchmark metrics do not establish domain deployment validity"],
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return payload


def _features(dataset: NativeDataset, train: np.ndarray, validation: np.ndarray, test: np.ndarray) -> tuple[object, object, object]:
    if dataset.modality == "text":
        from sklearn.feature_extraction.text import TfidfVectorizer

        vectorizer = TfidfVectorizer(max_features=1024, min_df=2, sublinear_tf=True)
        x_train = vectorizer.fit_transform([dataset.values[int(index)] for index in train])
        x_validation = vectorizer.transform([dataset.values[int(index)] for index in validation])
        x_test = vectorizer.transform([dataset.values[int(index)] for index in test])
        return x_train, x_validation, x_test
    values = np.asarray(dataset.values, dtype=np.float32)
    if dataset.modality == "image":
        n = len(values)
        values = values.reshape(n, 4, 7, 4, 7).mean(axis=(2, 4)).reshape(n, 16)
    elif dataset.modality == "timeseries":
        spectrum = np.abs(np.fft.rfft(values, axis=1))[:, 1:25]
        values = np.column_stack((spectrum, values.mean(axis=1), values.std(axis=1), values.min(axis=1), values.max(axis=1))).astype(np.float32)
    return values[train], values[validation], values[test]


def _stratified_cap(indices: np.ndarray, labels: np.ndarray, cap: int, seed: int) -> np.ndarray:
    if len(indices) <= cap:
        return np.asarray(indices)
    rng = np.random.default_rng(seed)
    classes = np.unique(labels[indices])
    class_indices = {class_id.item() if hasattr(class_id, "item") else class_id: indices[labels[indices] == class_id] for class_id in classes}
    exact = {class_id: cap * len(items) / len(indices) for class_id, items in class_indices.items()}
    quotas = {class_id: min(len(class_indices[class_id]), int(value)) for class_id, value in exact.items()}
    remaining = cap - sum(quotas.values())
    remainder_order = sorted(
        class_indices,
        key=lambda class_id: (exact[class_id] - quotas[class_id], len(class_indices[class_id]), repr(class_id)),
        reverse=True,
    )
    for class_id in remainder_order:
        if remaining == 0:
            break
        if quotas[class_id] < len(class_indices[class_id]):
            quotas[class_id] += 1
            remaining -= 1

    selected: list[int] = []
    for class_id, candidates in class_indices.items():
        selected.extend(rng.choice(candidates, size=quotas[class_id], replace=False).tolist())
    rng.shuffle(selected)
    return np.asarray(selected, dtype=int)


def _evaluation_sample(
    indices: np.ndarray,
    labels: np.ndarray,
    predictions: np.ndarray,
    confidence: np.ndarray,
    count: int,
    seed: int,
) -> list[int]:
    """Freeze a cohort without preferring errors or a convenient confidence range."""
    correctness = predictions == labels
    confidence_band = np.digitize(confidence, (0.5, 0.7, 0.9), right=True)
    strata = np.asarray(
        [f"{int(label)}:{int(is_correct)}:{int(band)}" for label, is_correct, band in zip(labels, correctness, confidence_band)]
    )
    positions = np.arange(len(indices))
    sampled_positions = _stratified_cap(positions, strata, min(count, len(indices)), seed + 701)
    return [int(indices[position]) for position in sampled_positions]


def _ece(probabilities: np.ndarray, labels: np.ndarray, bins: int = 10) -> float:
    confidence = probabilities.max(axis=1)
    correctness = probabilities.argmax(axis=1) == labels
    edges = np.linspace(0.0, 1.0, bins + 1)
    result = 0.0
    for lower, upper in zip(edges[:-1], edges[1:]):
        mask = (confidence > lower) & (confidence <= upper)
        if mask.any():
            result += mask.mean() * abs(float(correctness[mask].mean()) - float(confidence[mask].mean()))
    return float(result)


def _subgroup_metrics(labels: np.ndarray, predictions: np.ndarray, rare_classes: Sequence[int]) -> dict[str, object]:
    result: dict[str, object] = {}
    for name, mask in (
        ("rare", np.isin(labels, rare_classes)),
        ("common", ~np.isin(labels, rare_classes)),
    ):
        if not mask.any():
            result[name] = {"n_objects": 0, "accuracy": None, "macro_f1": None}
            continue
        from sklearn.metrics import accuracy_score, f1_score

        result[name] = {
            "n_objects": int(mask.sum()),
            "accuracy": float(accuracy_score(labels[mask], predictions[mask])),
            "macro_f1": float(f1_score(labels[mask], predictions[mask], average="macro")),
        }
    return result


def _error_taxonomy(
    labels: np.ndarray,
    predictions: np.ndarray,
    probabilities: np.ndarray,
    rare_classes: Sequence[int],
) -> dict[str, object]:
    wrong = predictions != labels
    confidence = probabilities.max(axis=1)
    pairs = Counter((int(truth), int(prediction)) for truth, prediction in zip(labels[wrong], predictions[wrong]))
    return {
        "total_errors": int(wrong.sum()),
        "high_confidence_errors": int(np.sum(wrong & (confidence >= 0.80))),
        "low_confidence_errors": int(np.sum(wrong & (confidence < 0.60))),
        "rare_class_errors": int(np.sum(wrong & np.isin(labels, rare_classes))),
        "top_confusions": [
            {"true_class": truth, "predicted_class": prediction, "count": count}
            for (truth, prediction), count in pairs.most_common(10)
        ],
    }


def _evaluate_explanation_pairs(
    model: object,
    x_train: object,
    x_test: object,
    test_indices: np.ndarray,
    evaluation_ids: Sequence[int],
    modality: str,
) -> list[dict[str, object]]:
    positions = {int(object_id): position for position, object_id in enumerate(test_indices)}
    selected_positions = [positions[object_id] for object_id in evaluation_ids]
    samples = _dense_rows(x_test, selected_positions)
    reference = _dense_reference(x_train)
    probabilities = np.asarray(model.predict_proba(samples), dtype=float)
    predicted_classes = probabilities.argmax(axis=1)
    coefficients = _linear_coefficients(model)
    if coefficients is None:
        return []
    transformed = _linear_space(model, samples)
    pairs = []
    method = {
        "tabular": "linear_native_contribution",
        "image": "pooled_pixel_contribution",
        "text": "tfidf_token_contribution",
        "timeseries": "spectral_window_contribution",
    }[modality]
    for row, (object_id, predicted_class) in enumerate(zip(evaluation_ids, predicted_classes)):
        attribution = transformed[row] * coefficients[int(predicted_class)]
        top_k = np.argsort(np.abs(attribution))[-min(10, len(attribution)) :]
        deleted = samples[row].copy()
        deleted[top_k] = reference[top_k]
        inserted = reference.copy()
        inserted[top_k] = samples[row, top_k]
        base_probability = float(probabilities[row, predicted_class])
        deleted_probability = float(model.predict_proba(deleted.reshape(1, -1))[0, predicted_class])
        inserted_probability = float(model.predict_proba(inserted.reshape(1, -1))[0, predicted_class])
        deletion = base_probability - deleted_probability
        insertion = inserted_probability - float(model.predict_proba(reference.reshape(1, -1))[0, predicted_class])
        attribution_hash = hashlib.sha256(np.ascontiguousarray(attribution).tobytes()).hexdigest()
        pairs.append(
            {
                "object_id": str(object_id),
                "method": method,
                "base_fidelity": deletion,
                "wrapped_fidelity": deletion,
                "metrics": {
                    "deletion": deletion,
                    "insertion": insertion,
                    "rank_agreement": 1.0,
                    "top_k": len(top_k),
                    "class_conditional": int(predicted_class),
                },
                "base_attribution_sha256": attribution_hash,
                "wrapped_attribution_sha256": attribution_hash,
                "same_model_object_reference_budget_seed": True,
            }
        )
    return pairs


def _linear_coefficients(model: object) -> np.ndarray | None:
    estimator = model
    named_steps = getattr(model, "named_steps", None)
    if named_steps:
        estimator = list(named_steps.values())[-1]
    coefficients = getattr(estimator, "coef_", None)
    if coefficients is None:
        return None
    values = np.asarray(coefficients, dtype=float)
    if len(values) == 1:
        values = np.vstack((-values[0], values[0]))
    return values


def _linear_space(model: object, samples: np.ndarray) -> np.ndarray:
    named_steps = getattr(model, "named_steps", None)
    if not named_steps or len(named_steps) == 1:
        return samples
    transformer = list(named_steps.values())[0]
    return np.asarray(transformer.transform(samples), dtype=float)


def _dense_rows(matrix: object, positions: Sequence[int]) -> np.ndarray:
    selected = matrix[positions]
    toarray = getattr(selected, "toarray", None)
    return np.asarray(toarray() if toarray else selected, dtype=float)


def _dense_reference(matrix: object) -> np.ndarray:
    mean = matrix.mean(axis=0)
    return np.asarray(mean).reshape(-1).astype(float)


def _required_method_coverage(modality: str, pairs: Sequence[Mapping[str, object]]) -> list[dict[str, str]]:
    required = {
        "tabular": ("SHAP", "LIME", "Anchors", "RuleFit"),
        "image": ("Grad-CAM", "Integrated Gradients"),
        "text": ("token masking",),
        "timeseries": ("window masking",),
    }[modality]
    measured = {str(row["method"]).lower() for row in pairs}
    return [
        {
            "method": method,
            "status": "measured" if method.lower().replace(" ", "_") in measured else "not_measured",
        }
        for method in required
    ]


def _hash(*arrays: np.ndarray) -> str:
    digest = hashlib.sha256()
    for array in arrays:
        digest.update(np.ascontiguousarray(array).tobytes())
    return digest.hexdigest()
