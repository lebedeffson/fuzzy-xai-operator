"""Read-only lifecycle observer: supervised, unsupervised and missing evidence.

Assertions deliberately verify *observable contracts*, not model correctness.
"""

from __future__ import annotations

import json
import subprocess
import sys
from dataclasses import replace

import numpy as np
from fuzzyxai.ml_observer import MLRunEvidence, observe_ml_run
from sklearn.cluster import KMeans
from sklearn.datasets import load_diabetes, load_iris
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split


def _status(report, code: str) -> str:
    matching = [item.status for item in report.checks if item.code == code]
    assert len(matching) == 1, (code, matching)
    return matching[0]


def _example() -> MLRunEvidence:
    dataset = load_iris()
    train, evaluation = train_test_split(
        np.arange(len(dataset.target)), test_size=0.25, random_state=42, stratify=dataset.target
    )
    fitted = RandomForestClassifier(n_estimators=20, random_state=42).fit(dataset.data[train], dataset.target[train])
    return MLRunEvidence(
        task="classification",
        model_family="random_forest",
        train_features=dataset.data[train],
        evaluation_features=dataset.data[evaluation],
        train_targets=dataset.target[train],
        train_feature_names=tuple(str(x) for x in dataset.feature_names),
        evaluation_feature_names=tuple(str(x) for x in dataset.feature_names),
        evaluation_targets=dataset.target[evaluation],
        predictions=fitted.predict(dataset.data[evaluation]),
        probabilities=fitted.predict_proba(dataset.data[evaluation]),
        probability_classes=tuple(str(x) for x in fitted.classes_),
        train_ids=tuple(f"row-{i}" for i in train),
        evaluation_ids=tuple(f"row-{i}" for i in evaluation),
        model_fit_ids=tuple(f"row-{i}" for i in train),
        preprocessor_fit_ids=tuple(f"row-{i}" for i in train),
        model_version="iris-rf-v1",
        random_seed=42,
    )


def test_classification_scores_without_train_or_prediction_mutation() -> None:
    example = _example()
    original_x = np.asarray(example.evaluation_features).copy()
    report = observe_ml_run(example)
    assert _status(report, "split_isolation") == "ok"
    assert _status(report, "ordered_feature_schema") == "ok"
    assert _status(report, "model_fit_isolation") == "ok"
    assert _status(report, "preprocessor_fit_isolation") == "ok"
    assert _status(report, "class_mapping") == "ok"
    assert report.metrics["accuracy"] >= 0.8
    assert report.metrics["macro_f1"] >= 0.8
    np.testing.assert_array_equal(example.evaluation_features, original_x)
    assert isinstance(report.to_dict(), dict)


def test_known_data_leakage_and_bad_probability_rows_are_problems() -> None:
    example = _example()
    assert example.train_ids is not None
    assert example.evaluation_ids is not None
    contaminated = replace(
        example,
        preprocessor_fit_ids=example.train_ids + example.evaluation_ids[:2],
        probabilities=np.asarray(example.probabilities) * 0.1,
    )
    report = observe_ml_run(contaminated)
    assert _status(report, "preprocessor_fit_isolation") == "problem"
    assert _status(report, "probabilities") == "problem"
    assert report.overall_status == "observed_with_problems"


def test_label_disagreement_is_warning_not_false_proof_of_error() -> None:
    example = _example()
    report = observe_ml_run(replace(example, predictions=np.zeros(len(example.predictions), dtype=int)))
    assert _status(report, "class_mapping") == "warning"
    assert "accuracy" in report.metrics


def test_missing_sources_explicitly_unknown_and_not_passed() -> None:
    report = observe_ml_run(MLRunEvidence(task="classification", model_family="external_model"))
    assert _status(report, "split_isolation") == "unknown"
    assert _status(report, "model_fit_isolation") == "unknown"
    assert _status(report, "probabilities") == "unknown"
    assert _status(report, "label_metrics") == "unknown"
    assert report.overall_status == "incomplete_evidence"


def test_nonfinite_features_detected_without_training() -> None:
    item = replace(_example(), evaluation_features=np.array([[np.nan] * 4]))
    report = observe_ml_run(item)
    assert _status(report, "evaluation_features") == "problem"


def test_regression_metrics_and_not_applicable_probs() -> None:
    dataset = load_diabetes()
    train, evaluation = train_test_split(np.arange(len(dataset.target)), test_size=0.2, random_state=7)
    model = LinearRegression().fit(dataset.data[train], dataset.target[train])
    report = observe_ml_run(MLRunEvidence(
        task="regression", model_family="linear_regression", train_features=dataset.data[train],
        evaluation_features=dataset.data[evaluation], evaluation_targets=dataset.target[evaluation],
        predictions=model.predict(dataset.data[evaluation]),
    ))
    assert _status(report, "probabilities") == "not_applicable"
    assert _status(report, "label_metrics") == "ok"
    assert report.metrics["mae"] >= 0
    assert report.metrics["rmse"] >= 0


def test_unsupervised_clustering_not_mistaken_for_classifier() -> None:
    dataset = load_iris()
    fitted = KMeans(n_clusters=3, random_state=42, n_init=10).fit(dataset.data)
    report = observe_ml_run(MLRunEvidence(
        task="clustering", model_family="kmeans",
        evaluation_features=dataset.data, predictions=fitted.labels_,
    ))
    assert _status(report, "probabilities") == "not_applicable"
    assert _status(report, "label_metrics") == "not_applicable"
    assert report.metrics["clusters_observed"] == 3


def test_duplicate_identifiers_cannot_be_reported_as_split_isolated() -> None:
    item = _example()
    assert item.evaluation_ids is not None
    assert item.train_ids is not None
    duplicated = replace(item, evaluation_ids=(item.train_ids[0],) + item.evaluation_ids[1:])
    report = observe_ml_run(duplicated)
    assert _status(report, "split_isolation") == "problem"


def test_unsupported_task_cannot_claim_validity() -> None:
    item = MLRunEvidence(task="generative", model_family="llm")  # type: ignore[arg-type]
    try:
        observe_ml_run(item)
    except ValueError as exc:
        assert "supported" in str(exc)
    else:
        raise AssertionError("Unsupported task should not be silently accepted")



def test_feature_order_swap_is_observable_even_with_equal_column_count() -> None:
    example = _example()
    assert example.evaluation_feature_names is not None
    swapped_names = list(example.evaluation_feature_names)
    swapped_names[0], swapped_names[1] = swapped_names[1], swapped_names[0]
    report = observe_ml_run(replace(example, evaluation_feature_names=tuple(swapped_names)))
    assert _status(report, "feature_width") == "ok"
    assert _status(report, "ordered_feature_schema") == "problem"


def test_unknown_feature_schema_is_not_reported_as_verified() -> None:
    report = observe_ml_run(replace(_example(), train_feature_names=None))
    assert _status(report, "ordered_feature_schema") == "unknown"



def test_json_cli_reads_evidence_and_writes_report_without_model_execution(tmp_path) -> None:
    payload = tmp_path / "evidence.json"
    output = tmp_path / "report.json"
    payload.write_text(
        json.dumps({
            "task": "regression",
            "model_family": "externally_fitted_linear",
            "evaluation_features": [[1.0], [2.0]],
            "predictions": [1.2, 2.3],
            "evaluation_targets": [1.0, 2.0],
        }),
        encoding="utf-8",
    )
    subprocess.run(
        [sys.executable, "-m", "fuzzyxai.ml_observer", str(payload), "--output", str(output)],
        check=True,
        capture_output=True,
        text=True,
    )
    result = json.loads(output.read_text(encoding="utf-8"))
    assert result["task"] == "regression"
    assert result["metrics"]["mae"] > 0
    assert result["overall_status"] == "incomplete_evidence"
    assert result["checks"]



def test_exact_duplicate_data_values_detected_even_with_new_row_ids() -> None:
    sample = _example()
    copied = np.asarray(sample.evaluation_features).copy()
    copied[0] = np.asarray(sample.train_features)[0]
    report = observe_ml_run(replace(sample, evaluation_features=copied))
    assert _status(report, "split_isolation") == "ok"
    assert _status(report, "cross_split_duplicate_rows") == "warning"
    assert report.metrics["cross_split_duplicate_rows"] >= 1


def test_duplicate_scan_not_applicable_to_missing_numeric_matrices() -> None:
    sample = _example()
    report = observe_ml_run(replace(sample, evaluation_features=None))
    assert _status(report, "cross_split_duplicate_rows") == "unknown"
