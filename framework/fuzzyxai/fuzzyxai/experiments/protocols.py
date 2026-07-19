"""Measured E1/E2 protocols for the full empirical-validation program."""

from __future__ import annotations

import csv
import hashlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from time import perf_counter
from typing import Any, Mapping, Sequence

import numpy as np

from .datasets import BenchmarkDataset, build_all_controlled, snapshot_dataset
from .metrics import binary_classification_metrics
from .statistics import holm_adjust, mcnemar_exact, paired_summary


@dataclass(frozen=True)
class ModelRun:
    dataset_id: str
    modality: str
    model_id: str
    model_family: str
    n_train: int
    n_test: int
    fit_seconds: float
    predict_seconds: float
    metrics: Mapping[str, float | int]
    available_channels: tuple[str, ...]
    native_channels: tuple[str, ...]
    surrogate_channels: tuple[str, ...]
    status: str = "measured"

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def _write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _write_csv(path: Path, rows: Sequence[Mapping[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        raise ValueError(f"refusing to write empty CSV: {path}")
    fields: list[str] = []
    for row in rows:
        for field in row:
            if field not in fields:
                fields.append(field)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        for row in rows:
            writer.writerow({key: _csv_value(row.get(key)) for key in fields})


def _csv_value(value: object) -> object:
    if isinstance(value, (dict, list, tuple)):
        return json.dumps(value, ensure_ascii=False, sort_keys=True)
    return value


def _image_features(images: np.ndarray) -> np.ndarray:
    matrix = np.asarray(images, dtype=float)
    n_objects, height, width = matrix.shape
    quadrants = np.column_stack(
        (
            matrix[:, : height // 2, : width // 2].mean(axis=(1, 2)),
            matrix[:, : height // 2, width // 2 :].mean(axis=(1, 2)),
            matrix[:, height // 2 :, : width // 2].mean(axis=(1, 2)),
            matrix[:, height // 2 :, width // 2 :].mean(axis=(1, 2)),
        )
    )
    rows = matrix.mean(axis=2)
    columns = matrix.mean(axis=1)
    return np.column_stack((matrix.reshape(n_objects, -1), quadrants, rows, columns))


def _time_series_features(values: np.ndarray) -> np.ndarray:
    matrix = np.asarray(values, dtype=float)
    spectrum = np.abs(np.fft.rfft(matrix, axis=1))[:, 1:9]
    differences = np.diff(matrix, axis=1)
    summary = np.column_stack(
        (
            matrix.mean(axis=1),
            matrix.std(axis=1),
            matrix.min(axis=1),
            matrix.max(axis=1),
            differences.mean(axis=1),
            differences.std(axis=1),
        )
    )
    return np.column_stack((matrix, spectrum, summary))


def _dataset_split(labels: np.ndarray, *, seed: int) -> tuple[np.ndarray, np.ndarray]:
    from sklearn.model_selection import train_test_split

    indices = np.arange(len(labels))
    train, test = train_test_split(indices, test_size=0.25, random_state=seed, stratify=labels)
    return np.sort(train), np.sort(test)


def _models_for(dataset: BenchmarkDataset, *, seed: int) -> list[tuple[str, str, Any, tuple[str, ...], tuple[str, ...]]]:
    from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.neural_network import MLPClassifier
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    if dataset.modality == "text":
        from sklearn.feature_extraction.text import TfidfVectorizer

        return [
            (
                "tfidf_logistic",
                "linear",
                make_pipeline(TfidfVectorizer(max_features=2000, ngram_range=(1, 2)), LogisticRegression(max_iter=500, random_state=seed)),
                ("token_importance", "token_masking", "provenance"),
                ("linear_coefficients",),
            ),
            (
                "tfidf_mlp",
                "neural_feature_model",
                make_pipeline(
                    TfidfVectorizer(max_features=800),
                    MLPClassifier(hidden_layer_sizes=(24,), max_iter=80, early_stopping=True, random_state=seed),
                ),
                ("token_importance", "token_masking", "stability", "provenance"),
                (),
            ),
        ]
    if dataset.modality == "image":
        return [
            (
                "image_logistic",
                "linear",
                make_pipeline(StandardScaler(), LogisticRegression(max_iter=500, random_state=seed)),
                ("perturbation_map", "prediction_preservation", "map_stability", "provenance"),
                ("linear_coefficients",),
            ),
            (
                "image_mlp",
                "neural_feature_model",
                MLPClassifier(hidden_layer_sizes=(32,), max_iter=80, early_stopping=True, random_state=seed),
                ("perturbation_map", "prediction_preservation", "map_stability", "provenance"),
                (),
            ),
        ]
    if dataset.modality == "time_series":
        return [
            (
                "window_hist_gradient_boosting",
                "gradient_boosting",
                HistGradientBoostingClassifier(max_iter=80, random_state=seed),
                ("important_intervals", "window_ablation", "noise_stability", "shift_stability", "provenance"),
                (),
            ),
            (
                "window_mlp",
                "neural_feature_model",
                make_pipeline(
                    StandardScaler(),
                    MLPClassifier(hidden_layer_sizes=(32,), max_iter=80, early_stopping=True, random_state=seed),
                ),
                ("important_intervals", "window_ablation", "noise_stability", "shift_stability", "provenance"),
                (),
            ),
        ]
    return [
        (
            "logistic_regression",
            "linear",
            make_pipeline(StandardScaler(), LogisticRegression(max_iter=500, random_state=seed)),
            ("feature_contributions", "similar_cases", "sensitivity", "group_metrics", "provenance"),
            ("linear_coefficients",),
        ),
        (
            "random_forest",
            "tree_ensemble",
            RandomForestClassifier(n_estimators=80, min_samples_leaf=3, n_jobs=1, random_state=seed),
            ("feature_contributions", "tree_paths", "similar_cases", "sensitivity", "group_metrics", "provenance"),
            ("tree_paths", "feature_importances"),
        ),
        (
            "hist_gradient_boosting",
            "gradient_boosting",
            HistGradientBoostingClassifier(max_iter=100, random_state=seed),
            ("feature_contributions", "similar_cases", "sensitivity", "group_metrics", "provenance"),
            (),
        ),
    ]


def _model_values(dataset: BenchmarkDataset) -> np.ndarray | tuple[str, ...]:
    if dataset.modality == "image":
        return _image_features(np.asarray(dataset.values))
    if dataset.modality == "time_series":
        return _time_series_features(np.asarray(dataset.values))
    return dataset.values


def run_multimodal_validation(
    *,
    output_root: Path,
    n_objects: int,
    seed: int = 42,
) -> dict[str, object]:
    datasets = build_all_controlled(n_objects=n_objects)
    data_dir = output_root / "data"
    dataset_rows = [snapshot_dataset(dataset, data_dir) for dataset in datasets]
    runs: list[ModelRun] = []
    for dataset in datasets:
        values = _model_values(dataset)
        train, test = _dataset_split(dataset.labels, seed=seed)
        train_values = _select(values, train)
        test_values = _select(values, test)
        for model_id, family, model, channels, native in _models_for(dataset, seed=seed):
            start = perf_counter()
            model.fit(train_values, dataset.labels[train])
            fit_seconds = perf_counter() - start
            start = perf_counter()
            probabilities = np.asarray(model.predict_proba(test_values), dtype=float)[:, 1]
            predict_seconds = perf_counter() - start
            metrics = binary_classification_metrics(
                dataset.labels[test],
                probabilities,
                subgroup_mask=dataset.rare_subgroup_mask[test],
            )
            runs.append(
                ModelRun(
                    dataset_id=dataset.dataset_id,
                    modality=dataset.modality,
                    model_id=model_id,
                    model_family=family,
                    n_train=len(train),
                    n_test=len(test),
                    fit_seconds=fit_seconds,
                    predict_seconds=predict_seconds,
                    metrics=metrics,
                    available_channels=channels,
                    native_channels=native,
                    surrogate_channels=tuple(channel for channel in channels if channel not in native),
                )
            )
    payload = {
        "schema_version": "1.0",
        "experiment_id": "E1",
        "result_origin": "measured_on_controlled_datasets",
        "claim_scope": "multimodal protocol behavior; no external-domain generalization",
        "datasets": dataset_rows,
        "runs": [run.to_dict() for run in runs],
        "checks": {
            "four_modalities": len({row["modality"] for row in dataset_rows}) == 4,
            "minimum_objects": all(int(row["n_objects"]) >= n_objects for row in dataset_rows),
            "all_models_measured": all(run.status == "measured" for run in runs),
            "provenance_channel": all("provenance" in run.available_channels for run in runs),
        },
        "limitations": [
            "controlled datasets do not establish external-domain validity",
            "neural sequence and CNN/ONNX channels are verified in a separate optional-runtime job",
            "attribution maps are associational, not causal",
        ],
    }
    _write_json(output_root / "datasets_manifest.json", {"schema_version": "1.0", "datasets": dataset_rows})
    _write_json(output_root / "multimodal_results.json", payload)
    flat_rows = []
    for run in runs:
        row = {key: value for key, value in run.to_dict().items() if key != "metrics"}
        row.update(run.metrics)
        flat_rows.append(row)
    _write_csv(output_root / "multimodal_results.csv", flat_rows)
    return payload


def _select(values: np.ndarray | tuple[str, ...], indices: np.ndarray) -> np.ndarray | list[str]:
    if isinstance(values, tuple):
        return [values[int(index)] for index in indices]
    return np.asarray(values)[indices]


def _select_rule(values: np.ndarray, labels: np.ndarray) -> dict[str, float]:
    candidates: list[tuple[float, float, float]] = []
    for upper_quantile in (0.70, 0.80, 0.90):
        for lower_quantile in (0.10, 0.20, 0.30):
            upper = float(np.quantile(values[:, 0], upper_quantile))
            lower = float(np.quantile(values[:, 1], lower_quantile))
            active = (values[:, 0] > upper) & (values[:, 1] < lower)
            if not active.any():
                score = 0.0
            else:
                true_positive = int(np.sum(active & (labels == 1)))
                precision = true_positive / int(active.sum())
                recall = true_positive / max(1, int(np.sum(labels == 1)))
                score = 2.0 * precision * recall / max(1e-12, precision + recall)
            candidates.append((score, upper, lower))
    score, upper, lower = max(candidates, key=lambda item: (item[0], -item[1], item[2]))
    return {"feature_00_gt": upper, "feature_01_lt": lower, "training_f1": score}


def _rule_indicator(values: np.ndarray, rule: Mapping[str, float]) -> np.ndarray:
    return ((values[:, 0] > rule["feature_00_gt"]) & (values[:, 1] < rule["feature_01_lt"])).astype(float)


def run_repeated_rule_ablation(
    *,
    output_root: Path,
    n_objects: int,
    folds: int,
    seeds: int,
    seed: int = 42,
) -> dict[str, object]:
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import StratifiedKFold
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    dataset = build_all_controlled(n_objects=n_objects)[0]
    values = np.asarray(dataset.values, dtype=float)
    metric_rows: list[dict[str, object]] = []
    prediction_rows: list[dict[str, object]] = []
    metric_names = ("accuracy", "balanced_accuracy", "precision", "recall", "f1", "auroc", "expected_calibration_error", "subgroup_recall", "subgroup_f1", "critical_errors")
    for seed_offset in range(seeds):
        split_seed = seed + seed_offset
        splitter = StratifiedKFold(n_splits=folds, shuffle=True, random_state=split_seed)
        for fold_index, (train, test) in enumerate(splitter.split(values, dataset.labels)):
            rule = _select_rule(values[train], dataset.labels[train])
            train_rule = _rule_indicator(values[train], rule)[:, None]
            test_rule = _rule_indicator(values[test], rule)[:, None]
            with_rule = make_pipeline(StandardScaler(), LogisticRegression(max_iter=500, random_state=split_seed))
            without_rule = make_pipeline(StandardScaler(), LogisticRegression(max_iter=500, random_state=split_seed))
            with_rule.fit(np.column_stack((values[train], train_rule)), dataset.labels[train])
            without_rule.fit(values[train], dataset.labels[train])
            probabilities_with = with_rule.predict_proba(np.column_stack((values[test], test_rule)))[:, 1]
            probabilities_without = without_rule.predict_proba(values[test])[:, 1]
            metrics_with = binary_classification_metrics(
                dataset.labels[test], probabilities_with, subgroup_mask=dataset.rare_subgroup_mask[test]
            )
            metrics_without = binary_classification_metrics(
                dataset.labels[test], probabilities_without, subgroup_mask=dataset.rare_subgroup_mask[test]
            )
            row: dict[str, object] = {
                "repeat_seed": split_seed,
                "fold": fold_index,
                "n_train": len(train),
                "n_test": len(test),
                "rule_selected_on": "train_only",
                "rule": rule,
                "rule_test_coverage": float(test_rule.mean()),
            }
            for name in metric_names:
                row[f"with_rule_{name}"] = metrics_with[name]
                row[f"without_rule_{name}"] = metrics_without[name]
            metric_rows.append(row)
            predictions_with = (probabilities_with >= 0.5).astype(int)
            predictions_without = (probabilities_without >= 0.5).astype(int)
            for position, object_index in enumerate(test):
                prediction_rows.append(
                    {
                        "repeat_seed": split_seed,
                        "fold": fold_index,
                        "object_id": str(dataset.object_ids[object_index]),
                        "true_label": int(dataset.labels[object_index]),
                        "rare_subgroup": bool(dataset.rare_subgroup_mask[object_index]),
                        "critical": bool(dataset.critical_mask[object_index]),
                        "rule_active": bool(test_rule[position, 0]),
                        "probability_with_rule": float(probabilities_with[position]),
                        "probability_without_rule": float(probabilities_without[position]),
                        "prediction_with_rule": int(predictions_with[position]),
                        "prediction_without_rule": int(predictions_without[position]),
                    }
                )
    statistical: dict[str, object] = {}
    p_values: list[float] = []
    for metric_name in metric_names:
        higher_is_better = metric_name not in {"expected_calibration_error", "critical_errors"}
        summary = paired_summary(
            [float(row[f"with_rule_{metric_name}"]) for row in metric_rows],
            [float(row[f"without_rule_{metric_name}"]) for row in metric_rows],
            higher_is_better=higher_is_better,
            seed=seed,
        ).to_dict()
        statistical[metric_name] = summary
        p_values.append(float(summary["wilcoxon_p_two_sided"]))
    adjusted = holm_adjust(p_values)
    for metric_name, adjusted_value in zip(metric_names, adjusted):
        assert isinstance(statistical[metric_name], dict)
        statistical[metric_name]["holm_adjusted_p"] = adjusted_value  # type: ignore[index]
    correctness_with = [row["prediction_with_rule"] == row["true_label"] for row in prediction_rows]
    correctness_without = [row["prediction_without_rule"] == row["true_label"] for row in prediction_rows]
    report = {
        "schema_version": "1.0",
        "experiment_id": "E2",
        "result_origin": "measured_on_controlled_tabular_dataset",
        "n_paired_comparisons": len(metric_rows),
        "folds": folds,
        "independent_seeds": seeds,
        "test_partition_used_for_rule_selection": False,
        "statistics": statistical,
        "mcnemar": mcnemar_exact(correctness_with, correctness_without),
        "interpretation": _ablation_interpretation(statistical),
        "limitations": ["controlled interaction subgroup", "rule-selection family is fixed before evaluation"],
    }
    _write_csv(output_root / "repeated_cv_metrics.csv", metric_rows)
    _write_json(output_root / "statistical_report.json", report)
    _write_object_predictions(output_root, prediction_rows)
    return report


def _write_object_predictions(output_root: Path, rows: Sequence[Mapping[str, object]]) -> None:
    _write_csv(output_root / "repeated_cv_predictions.csv", rows)
    try:
        import pandas as pd

        pd.DataFrame(rows).to_parquet(output_root / "repeated_cv_predictions.parquet", index=False)
    except (ImportError, ModuleNotFoundError) as error:
        _write_json(
            output_root / "repeated_cv_predictions.parquet.unavailable.json",
            {"status": "not_available", "reason": type(error).__name__, "csv_fallback": "repeated_cv_predictions.csv"},
        )


def _ablation_interpretation(statistical: Mapping[str, object]) -> str:
    subgroup = statistical["subgroup_recall"]
    global_recall = statistical["recall"]
    assert isinstance(subgroup, Mapping) and isinstance(global_recall, Mapping)
    subgroup_effect = float(subgroup["mean_difference"])
    global_effect = float(global_recall["mean_difference"])
    stable = float(subgroup["worsening_fraction"]) >= 0.8 and float(subgroup["holm_adjusted_p"]) <= 0.05
    if stable and subgroup_effect < global_effect:
        return "rule removal systematically harms the rare-group metric more than the global metric"
    return "the rule-removal effect is not confirmed as a general pattern and remains a controlled illustration"


def hash_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()
