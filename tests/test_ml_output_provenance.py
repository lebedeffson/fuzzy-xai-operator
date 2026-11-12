"""Real output-call provenance tests; the observer itself does not infer."""

from __future__ import annotations

import json
from dataclasses import asdict, replace

import numpy as np
import pytest
from fuzzyxai.ml_observer import MLRunEvidence, observe_ml_run
from fuzzyxai.output_provenance import OutputRecorder, audit_output_calls
from sklearn.datasets import load_diabetes, load_iris
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier


def _classification():
    data = load_iris()
    idx_train, idx_test = train_test_split(
        np.arange(len(data.target)), random_state=119, test_size=0.25,
        stratify=data.target,
    )
    x_train, x_eval = data.data[idx_train].copy(), data.data[idx_test].copy()
    y_train, y_eval = data.target[idx_train], data.target[idx_test]
    ids = tuple(f"iris:{i}" for i in idx_test)
    model = DecisionTreeClassifier(random_state=119).fit(x_train, y_train)
    recorder = OutputRecorder()
    prediction = recorder.predict(model, x_eval, row_ids=ids)
    probability = recorder.predict_proba(model, x_eval, row_ids=ids)
    evidence = MLRunEvidence(
        task="classification", model_family=type(model).__name__,
        train_features=x_train, evaluation_features=x_eval,
        train_targets=y_train, evaluation_targets=y_eval,
        predictions=prediction, probabilities=probability,
        probability_classes=tuple(str(c) for c in model.classes_),
        evaluation_ids=ids,
        output_records=tuple(item.to_dict() for item in recorder.records),
    )
    return model, recorder, evidence


def _status(report, code):
    results = [entry.status for entry in report.checks if entry.code == code]
    assert len(results) == 1, (code, results)
    return results[0]


def test_observed_predict_and_proba_match_report_without_inference(monkeypatch) -> None:
    model, recorder, example = _classification()
    original = np.asarray(example.evaluation_features).copy()

    def forbidden(*args, **kwargs):
        raise AssertionError("Observer must never call inference")

    monkeypatch.setattr(model, "predict", forbidden)
    monkeypatch.setattr(model, "predict_proba", forbidden)
    report = observe_ml_run(example)
    for suffix in (
        "output_trace_predict_row_ids",
        "output_trace_predict_inputs",
        "output_trace_predict_values",
        "output_trace_predict_proba_row_ids",
        "output_trace_predict_proba_inputs",
        "output_trace_predict_proba_values",
        "output_trace_predict_proba_classes",
    ):
        assert _status(report, suffix) == "ok"
    assert len(recorder.records) == 2
    np.testing.assert_array_equal(original, example.evaluation_features)
    assert "accuracy" in report.metrics


def test_swapped_prediction_rows_change_witness_even_if_shape_and_class_count_match() -> None:
    _, _, example = _classification()
    corrupted = np.asarray(example.predictions).copy()
    a, b = next(
        (a, b)
        for a in range(len(corrupted))
        for b in range(a + 1, len(corrupted))
        if corrupted[a] != corrupted[b]
    )
    corrupted[a], corrupted[b] = corrupted[b], corrupted[a]
    report = observe_ml_run(replace(example, predictions=corrupted))
    assert _status(report, "predictions") == "ok"
    assert _status(report, "output_trace_predict_values") == "problem"
    assert _status(report, "output_trace_predict_proba_values") == "ok"


def test_valid_probability_values_with_swapped_columns_fail_witness() -> None:
    _, _, example = _classification()
    probabilities = np.asarray(example.probabilities).copy()
    probabilities[:, [0, 1]] = probabilities[:, [1, 0]]
    altered_classes = tuple(example.probability_classes[i] for i in (1, 0, 2))
    report = observe_ml_run(replace(example, probabilities=probabilities, probability_classes=altered_classes))
    assert _status(report, "probabilities") == "ok"
    assert _status(report, "output_trace_predict_proba_values") == "problem"
    assert _status(report, "output_trace_predict_proba_classes") == "problem"


def test_reported_eval_input_row_swap_detected_without_changing_prediction_shape() -> None:
    _, _, example = _classification()
    changed = np.asarray(example.evaluation_features).copy()
    changed[[0, 1]] = changed[[1, 0]]
    report = observe_ml_run(replace(example, evaluation_features=changed))
    assert _status(report, "output_trace_predict_row_ids") == "ok"
    assert _status(report, "output_trace_predict_inputs") == "problem"
    assert _status(report, "output_trace_predict_values") == "ok"


def test_reported_ids_swapped_even_when_feature_values_match() -> None:
    _, _, example = _classification()
    changed = list(example.evaluation_ids)
    changed[0], changed[1] = changed[1], changed[0]
    report = observe_ml_run(replace(example, evaluation_ids=tuple(changed)))
    assert _status(report, "output_trace_predict_row_ids") == "problem"
    assert _status(report, "output_trace_predict_inputs") == "ok"


def test_absent_witness_does_not_claim_a_produced_prediction() -> None:
    _, _, example = _classification()
    report = observe_ml_run(replace(example, output_records=None))
    assert _status(report, "output_trace") == "unknown"


def test_invalid_record_digest_is_rejected() -> None:
    _, _, example = _classification()
    malformed = example.output_records[0].copy()
    malformed["input_batch_sha256"] = "a" * 64
    report = observe_ml_run(replace(example, output_records=[malformed]))
    assert _status(report, "output_trace") == "problem"


def test_multiple_output_calls_without_selection_do_not_appear_verified() -> None:
    model, recorder, example = _classification()
    recorder.predict(model, example.evaluation_features, row_ids=example.evaluation_ids)
    report = observe_ml_run(replace(
        example, output_records=tuple(item.to_dict() for item in recorder.records),
    ))
    assert _status(report, "output_trace_predict") == "unknown"
    assert _status(report, "output_trace_predict_proba_values") == "ok"


def test_transformed_feature_space_is_unknown_without_transform_evidence() -> None:
    model, _, example = _classification()
    recorder = OutputRecorder()
    x = np.asarray(example.evaluation_features)
    predictions = recorder.predict(model, x, row_ids=example.evaluation_ids, feature_space="transformed")
    report = observe_ml_run(replace(example, predictions=predictions, output_records=[recorder.records[0].to_dict()]))
    assert _status(report, "output_trace_predict_inputs") == "unknown"
    assert _status(report, "output_trace_predict_values") == "ok"


def test_inference_exception_does_not_create_falsely_successful_witness() -> None:
    class FailingModel:
        def predict(self, X):
            raise RuntimeError("inference failure")

    _, _, example = _classification()
    recorder = OutputRecorder()
    with pytest.raises(RuntimeError, match="inference failure"):
        recorder.predict(FailingModel(), example.evaluation_features, row_ids=example.evaluation_ids)
    assert recorder.records == ()


def test_regression_model_output_witness() -> None:
    source = load_diabetes()
    tr, ev = train_test_split(np.arange(len(source.target)), test_size=0.2, random_state=119)
    x_train, x_eval = source.data[tr], source.data[ev]
    model = LinearRegression().fit(x_train, source.target[tr])
    output = OutputRecorder()
    ids = tuple(f"diabetes:{index}" for index in ev)
    predictions = output.predict(model, x_eval, row_ids=ids)
    report = observe_ml_run(MLRunEvidence(
        task="regression", model_family="LinearRegression",
        train_features=x_train, evaluation_features=x_eval,
        evaluation_targets=source.target[ev], predictions=predictions,
        evaluation_ids=ids, output_records=[item.to_dict() for item in output.records],
    ))
    assert _status(report, "output_trace_predict_values") == "ok"
    assert _status(report, "output_trace_predict_inputs") == "ok"
    assert report.metrics["mae"] > 0


def test_string_label_predictions_are_reported_as_unavailable_not_falsified() -> None:
    source = load_iris()
    targets = np.asarray([f"cls_{int(x)}" for x in source.target])
    model = DecisionTreeClassifier(random_state=119).fit(source.data, targets)
    recorder = OutputRecorder()
    x = source.data[:8]
    ids = tuple(f"example:{i}" for i in range(len(x)))
    prediction = recorder.predict(model, x, row_ids=ids)
    assert recorder.records[0].output_status == "unavailable"
    report = audit_output_calls(
        recorder.records, evaluation_features=x, evaluation_ids=ids, predictions=prediction,
    )
    statuses = {entry.code: entry.status for entry in report}
    assert statuses["output_trace_predict_values"] == "unknown"
    assert statuses["output_trace_predict_inputs"] == "ok"


def test_report_remains_usable_after_json_serialization() -> None:
    _, _, example = _classification()
    original = asdict(example)
    for name in ("train_features", "evaluation_features", "train_targets", "evaluation_targets", "predictions", "probabilities"):
        original[name] = np.asarray(original[name]).tolist()
    recreated = MLRunEvidence(**json.loads(json.dumps(original)))
    result = observe_ml_run(recreated)
    assert _status(result, "output_trace_predict_values") == "ok"
    assert _status(result, "output_trace_predict_proba_classes") == "ok"


def test_honestly_witnessed_but_wrong_predictions_do_not_prove_model_quality() -> None:
    _, _, example = _classification()
    assert _status(observe_ml_run(example), "output_trace_predict_values") == "ok"
    # Provenance only says the report matches the model. It does NOT
    # certify that target labels, predictive accuracy or decisions are right.
    wrong_reference = np.roll(np.asarray(example.evaluation_targets), 1)
    outcome = observe_ml_run(replace(example, evaluation_targets=wrong_reference))
    assert _status(outcome, "output_trace_predict_values") == "ok"
    assert outcome.metrics["accuracy"] <= 1.0
