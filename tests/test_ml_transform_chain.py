"""Actual StandardScaler transform calls joined to fit/output witnesses."""

from __future__ import annotations

import json
from dataclasses import asdict, replace

import numpy as np
import pytest
from fuzzyxai.fit_provenance import FitRecorder
from fuzzyxai.ml_observer import MLRunEvidence, observe_ml_run
from fuzzyxai.output_provenance import OutputRecorder
from fuzzyxai.transform_provenance import TransformRecorder, audit_transform_chain
from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier

_SECRET = b"only-a-development-test-secret-2026"


def _setup():
    data = load_wine()
    train, evaluation = train_test_split(
        np.arange(len(data.target)), random_state=54, test_size=0.25,
        stratify=data.target,
    )
    raw_train, raw_eval = data.data[train].copy(), data.data[evaluation].copy()
    train_y, eval_y = data.target[train].copy(), data.target[evaluation].copy()
    train_ids = tuple(f"wine:{int(i)}" for i in train)
    eval_ids = tuple(f"wine:{int(i)}" for i in evaluation)

    fit = FitRecorder(target_hmac_key=_SECRET)
    scaler = fit.fit(StandardScaler(), raw_train, row_ids=train_ids, role="preprocessor")
    transformer = TransformRecorder()
    transformed_train = transformer.transform(scaler, raw_train, split="train", row_ids=train_ids)
    transformed_eval = transformer.transform(scaler, raw_eval, split="evaluation", row_ids=eval_ids)
    model = fit.fit(
        DecisionTreeClassifier(random_state=54), transformed_train, train_y,
        role="model", row_ids=train_ids, feature_space="transformed",
    )
    outputs = OutputRecorder()
    predictions = outputs.predict(model, transformed_eval, row_ids=eval_ids, feature_space="transformed")
    probabilities = outputs.predict_proba(model, transformed_eval, row_ids=eval_ids, feature_space="transformed")
    evidence = MLRunEvidence(
        task="classification", model_family=type(model).__name__,
        train_features=raw_train, evaluation_features=raw_eval,
        train_targets=train_y, evaluation_targets=eval_y,
        predictions=predictions, probabilities=probabilities,
        probability_classes=tuple(str(x) for x in model.classes_),
        train_ids=train_ids, evaluation_ids=eval_ids,
        model_fit_ids=train_ids, preprocessor_fit_ids=train_ids,
        preprocessor_applied=True,
        fit_records=[r.to_dict() for r in fit.records],
        output_records=[r.to_dict() for r in outputs.records],
        transform_records=[r.to_dict() for r in transformer.records],
    )
    return evidence, fit, transformer, scaler, model, transformed_train, transformed_eval


def _statuses(evidence) -> dict[str, str]:
    report = observe_ml_run(evidence, target_hmac_key=_SECRET)
    return {check.code: check.status for check in report.checks}


def test_full_pipeline_links_raw_source_preprocessor_fit_model_fit_and_inference(monkeypatch) -> None:
    evidence, _, _, scaler, model, _, _ = _setup()
    copy_train = np.asarray(evidence.train_features).copy()
    copy_eval = np.asarray(evidence.evaluation_features).copy()

    def forbidden(*args, **kwargs):
        raise AssertionError("Observer must never call fit, transform or predict")

    monkeypatch.setattr(scaler, "transform", forbidden)
    monkeypatch.setattr(model, "predict", forbidden)
    monkeypatch.setattr(model, "predict_proba", forbidden)
    result = _statuses(evidence)
    for code in (
        "transform_train_input", "transform_evaluation_input",
        "transform_identity", "transform_preprocessor_fit_link",
        "transform_model_fit_link", "transform_predict_link",
        "transform_predict_proba_link", "fit_trace_model_targets",
        "fit_trace_preprocessor_content", "output_trace_predict_values",
        "output_trace_predict_proba_values",
    ):
        assert result[code] == "ok", (code, result[code])
    # Older raw-only checks remain conservatively UNKNOWN even if the new
    # cross-stage transformed-input link succeeds.
    assert result["fit_trace_model_content"] == "unknown"
    assert result["output_trace_predict_inputs"] == "unknown"
    np.testing.assert_array_equal(evidence.train_features, copy_train)
    np.testing.assert_array_equal(evidence.evaluation_features, copy_eval)


def test_model_fit_on_other_transformed_values_is_detected() -> None:
    evidence, _, _, _, model, train_x, _ = _setup()
    record = FitRecorder(target_hmac_key=_SECRET)
    record.fit(model, train_x + 0.2, evidence.train_targets, role="model",
               row_ids=evidence.train_ids, feature_space="transformed")
    events = [evidence.fit_records[0], record.records[0].to_dict()]
    outcome = _statuses(replace(evidence, fit_records=events))
    assert outcome["transform_train_input"] == "ok"
    assert outcome["transform_model_fit_link"] == "problem"


def test_inference_on_modified_transformed_values_detected_even_when_shapes_valid() -> None:
    evidence, _, _, _, model, _, evaluation_x = _setup()
    output = OutputRecorder()
    predictions = output.predict(
        model, evaluation_x + 0.15,
        row_ids=evidence.evaluation_ids, feature_space="transformed",
    )
    stored = [output.records[0].to_dict(), evidence.output_records[1]]
    outcome = _statuses(replace(evidence, predictions=predictions, output_records=stored))
    assert outcome["transform_predict_link"] == "problem"
    assert outcome["transform_predict_proba_link"] == "ok"


def test_changed_raw_evaluation_source_is_not_certified_by_transform_witness() -> None:
    evidence, *_ = _setup()
    changed = np.asarray(evidence.evaluation_features).copy()
    changed[0, 0] += 0.2
    result = _statuses(replace(evidence, evaluation_features=changed))
    assert result["transform_evaluation_input"] == "problem"
    assert result["transform_train_input"] == "ok"


def test_missing_transform_events_remain_unknown_not_ok() -> None:
    evidence, *_ = _setup()
    result = _statuses(replace(evidence, transform_records=None))
    assert result["transform_trace"] == "unknown"
    assert result["fit_trace_model_content"] == "unknown"


def test_forged_digest_is_rejected_before_joining() -> None:
    evidence, *_ = _setup()
    events = [dict(e) for e in evidence.transform_records]
    events[0]["output_batch_sha256"] = "f" * 64
    checks = _statuses(replace(evidence, transform_records=events))
    assert checks["transform_trace"] == "problem"


def test_two_train_transform_events_without_selection_are_unknown() -> None:
    evidence, *_ = _setup()
    events = [dict(e) for e in evidence.transform_records]
    events.append(dict(events[0]))
    result = _statuses(replace(evidence, transform_records=events))
    assert result["transform_train_input"] == "unknown"
    assert result["transform_model_fit_link"] == "unknown"


def test_preprocessor_fitted_on_evaluation_inputs_is_caught() -> None:
    evidence, *_ = _setup()
    raw_combined = np.concatenate([evidence.train_features, evidence.evaluation_features])
    contaminated = FitRecorder()
    contaminated.fit(
        StandardScaler(), raw_combined, row_ids=evidence.train_ids + evidence.evaluation_ids,
        role="preprocessor",
    )
    combined = [contaminated.records[0].to_dict(), evidence.fit_records[1]]
    result = _statuses(replace(evidence, fit_records=combined))
    assert result["fit_trace_preprocessor_isolation"] == "problem"
    assert result["transform_preprocessor_fit_link"] == "problem"


def test_label_source_corruption_does_not_conflate_with_transform_link() -> None:
    evidence, *_ = _setup()
    corrupt_targets = np.roll(np.asarray(evidence.train_targets), 1)
    result = _statuses(replace(evidence, train_targets=corrupt_targets))
    assert result["fit_trace_model_targets"] == "problem"
    assert result["transform_model_fit_link"] == "ok"


def test_failed_transform_call_is_not_recorded() -> None:
    class FailingTransform:
        def transform(self, X):
            raise RuntimeError("intentional transform failure")

    evidence, *_ = _setup()
    records = TransformRecorder()
    with pytest.raises(RuntimeError, match="intentional transform failure"):
        records.transform(FailingTransform(), evidence.train_features,
                          split="train", row_ids=evidence.train_ids)
    assert records.records == ()


def test_no_preprocessing_is_not_applicable_but_contradictory_event_is_problem() -> None:
    evidence, *_ = _setup()
    empty = _statuses(replace(evidence, preprocessor_applied=False, transform_records=None))
    assert empty["transform_trace"] == "not_applicable"
    nonempty = _statuses(replace(evidence, preprocessor_applied=False))
    assert nonempty["transform_trace"] == "problem"


def test_recorded_pipeline_can_be_reopened_from_json_without_sklearn_model() -> None:
    evidence, *_ = _setup()
    serial = asdict(evidence)
    for name in ("train_features", "evaluation_features", "train_targets",
                 "evaluation_targets", "predictions", "probabilities"):
        serial[name] = np.asarray(serial[name]).tolist()
    recovered = MLRunEvidence(**json.loads(json.dumps(serial)))
    results = _statuses(recovered)
    assert results["transform_preprocessor_fit_link"] == "ok"
    assert results["transform_model_fit_link"] == "ok"
    assert results["transform_predict_link"] == "ok"


def test_raw_transform_output_unavailable_does_not_claim_verified_chain() -> None:
    class UnsupportedTransform:
        def transform(self, X):
            return np.asarray(X, dtype=str)

    evidence, *_ = _setup()
    recorder = TransformRecorder()
    result = recorder.transform(
        UnsupportedTransform(), evidence.evaluation_features,
        split="evaluation", row_ids=evidence.evaluation_ids,
    )
    assert np.asarray(result).dtype.kind == "U"
    assert recorder.records[0].output_status == "unavailable"
    checks = audit_transform_chain(
        [evidence.transform_records[0], recorder.records[0].to_dict()],
        train_features=evidence.train_features,
        evaluation_features=evidence.evaluation_features,
        train_ids=evidence.train_ids, evaluation_ids=evidence.evaluation_ids,
        fit_records=evidence.fit_records, output_records=evidence.output_records,
        preprocessor_applied=True, task="classification",
    )
    state = {item.code: item.status for item in checks}
    assert state["transform_predict_link"] == "unknown"
