"""Fit-call provenance is observed at explicit sklearn fit boundaries.

The observer never trains. These tests *explicitly* trigger fitting via
the caller-facing recorder, then inspect the immutable recorded evidence.
"""

from __future__ import annotations

import json
from dataclasses import asdict

import numpy as np
import pytest
from fuzzyxai.fit_provenance import FitRecorder, audit_fit_trace
from fuzzyxai.ml_observer import MLRunEvidence, observe_ml_run
from sklearn.datasets import load_iris
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


def _split():
    d = load_iris()
    x = np.asarray(d.data, dtype=float)
    y = np.asarray(d.target)
    tr, te = train_test_split(np.arange(len(y)), random_state=71, stratify=y, test_size=0.25)
    tr_ids, te_ids = tuple(f"iris-row-{i}" for i in tr), tuple(f"iris-row-{i}" for i in te)
    return x[tr].copy(), x[te].copy(), y[tr], y[te], tr_ids, te_ids


def _status(report, code: str) -> str:
    found = [check.status for check in report.checks if check.code == code]
    assert len(found) == 1, (code, found)
    return found[0]


def _evidence(recorder: FitRecorder, model: LogisticRegression, train_x, test_x, train_y, test_y, train_ids, test_ids, *, preprocessor: bool) -> MLRunEvidence:
    # For the prediction side, this is only used when model input is raw.
    return MLRunEvidence(
        task="classification",
        model_family="logistic_regression",
        train_features=train_x,
        evaluation_features=test_x,
        train_targets=train_y,
        evaluation_targets=test_y,
        predictions=model.predict(test_x),
        probabilities=model.predict_proba(test_x),
        probability_classes=tuple(str(c) for c in model.classes_),
        train_feature_names=("sepal_length", "sepal_width", "petal_length", "petal_width"),
        evaluation_feature_names=("sepal_length", "sepal_width", "petal_length", "petal_width"),
        train_ids=train_ids,
        evaluation_ids=test_ids,
        model_fit_ids=train_ids,
        preprocessor_applied=preprocessor,
        fit_records=tuple(record.to_dict() for record in recorder.records),
        random_seed=71,
        model_version="iris-observed-fit",
    )


def test_plain_sklearn_estimator_fits_once_and_observer_never_refits(monkeypatch) -> None:
    train_x, test_x, train_y, test_y, tr_ids, te_ids = _split()
    recorder = FitRecorder()
    fit_calls: list[int] = []
    original_fit = LogisticRegression.fit

    def counted_fit(self, X, y, **kwargs):
        fit_calls.append(len(X))
        return original_fit(self, X, y, **kwargs)

    monkeypatch.setattr(LogisticRegression, "fit", counted_fit)
    model = LogisticRegression(max_iter=700, random_state=71)
    result = recorder.fit(model, train_x, train_y, row_ids=tr_ids, role="model")
    assert result is model and fit_calls == [len(train_x)]
    evidence = _evidence(recorder, model, train_x, test_x, train_y, test_y, tr_ids, te_ids, preprocessor=False)
    report = observe_ml_run(evidence)
    assert fit_calls == [len(train_x)]  # Observer did not train or predict.
    assert _status(report, "fit_trace_model_isolation") == "ok"
    assert _status(report, "fit_trace_model_content") == "ok"
    assert _status(report, "fit_trace_preprocessor_isolation") == "not_applicable"
    assert _status(report, "fit_trace_model_declarations") == "ok"


def test_early_fit_failure_does_not_create_fake_successful_provenance() -> None:
    class BadEstimator:
        def fit(self, X, y):
            raise RuntimeError("fit failed")

    x, _, y, _, ids, _ = _split()
    recorder = FitRecorder()
    with pytest.raises(RuntimeError, match="fit failed"):
        recorder.fit(BadEstimator(), x, y, row_ids=ids, role="model")
    assert recorder.records == ()


def test_real_preprocessor_leak_is_recorded_even_with_clean_manual_metadata() -> None:
    x, test_x, y, test_y, tr_ids, te_ids = _split()
    recorder = FitRecorder()
    preprocessor = recorder.fit(
        StandardScaler(), np.vstack([x, test_x]), row_ids=tr_ids + te_ids,
        role="preprocessor", feature_space="raw",
    )
    model = recorder.fit(
        LogisticRegression(max_iter=700, random_state=71), preprocessor.transform(x),
        y, row_ids=tr_ids, role="model", feature_space="transformed",
    )
    evidence = MLRunEvidence(
        task="classification", model_family="logistic_regression",
        train_features=x, evaluation_features=test_x,
        train_targets=y, evaluation_targets=test_y,
        predictions=model.predict(preprocessor.transform(test_x)),
        probabilities=model.predict_proba(preprocessor.transform(test_x)),
        probability_classes=tuple(str(c) for c in model.classes_),
        train_ids=tr_ids, evaluation_ids=te_ids,
        model_fit_ids=tr_ids, preprocessor_fit_ids=tr_ids,  # Deliberately misleading metadata baseline.
        preprocessor_applied=True,
        fit_records=[asdict(r) for r in recorder.records],
    )
    report = observe_ml_run(evidence)
    assert _status(report, "preprocessor_fit_isolation") == "ok"  # Ordinary ID-based declaration passes.
    assert _status(report, "fit_trace_preprocessor_isolation") == "problem"  # Witness catches actual call.
    assert _status(report, "fit_trace_preprocessor_content") == "ok"  # Bytes match both supplied partitions.
    assert _status(report, "fit_trace_preprocessor_declarations") == "warning"
    assert _status(report, "fit_trace_model_isolation") == "ok"
    assert _status(report, "fit_trace_model_content") == "unknown"  # Standardized features not the raw matrix.


def test_false_row_ids_do_not_hide_mismatched_raw_fit_content() -> None:
    x, test_x, y, test_y, tr_ids, te_ids = _split()
    recorder = FitRecorder()
    subset = test_x
    mislabeled_ids = tr_ids[:len(test_x)]
    labels = test_y
    # A genuine fit call on evaluation feature values, but caller supplied
    # training IDs. Ordinary ID provenance alone cannot discover the mismatch.
    recorder.fit(LogisticRegression(max_iter=700), subset, labels, row_ids=mislabeled_ids, role="model")
    result = audit_fit_trace(
        recorder.records, train_features=x, evaluation_features=test_x,
        train_ids=tr_ids, evaluation_ids=te_ids,
    )
    statuses = {check.code: check.status for check in result}
    assert statuses["fit_trace_model_isolation"] == "ok"
    assert statuses["fit_trace_model_content"] == "problem"


def test_no_fitted_source_witness_is_unknown_not_success() -> None:
    x, test_x, _, _, tr_ids, te_ids = _split()
    report = observe_ml_run(MLRunEvidence(
        task="regression", model_family="unobserved", train_features=x,
        evaluation_features=test_x, train_ids=tr_ids, evaluation_ids=te_ids,
    ))
    assert _status(report, "fit_trace") == "unknown"


def test_tampered_digest_fails_and_does_not_establish_authenticity() -> None:
    x, test_x, y, _, tr_ids, te_ids = _split()
    rec = FitRecorder()
    rec.fit(LogisticRegression(max_iter=700), x, y, row_ids=tr_ids, role="model")
    tampered = asdict(rec.records[0])
    tampered["batch_sha256"] = "a" * 64
    checks = audit_fit_trace([tampered], train_features=x, evaluation_features=test_x, train_ids=tr_ids, evaluation_ids=te_ids)
    assert len(checks) == 1
    assert checks[0].code == "fit_trace" and checks[0].status == "problem"


def test_captured_evidence_roundtrips_through_public_json_structure() -> None:
    x, test_x, y, _, tr_ids, te_ids = _split()
    rec = FitRecorder()
    model = rec.fit(LogisticRegression(max_iter=700), x, y, row_ids=tr_ids, role="model")
    raw = MLRunEvidence(
        task="classification", model_family="logistic_regression",
        train_features=x.tolist(), evaluation_features=test_x.tolist(),
        predictions=model.predict(test_x).tolist(),
        train_ids=tr_ids, evaluation_ids=te_ids, fit_records=[r.to_dict() for r in rec.records],
        preprocessor_applied=False,
    )
    again = MLRunEvidence(**json.loads(json.dumps(asdict(raw))))
    report = observe_ml_run(again)
    assert _status(report, "fit_trace_model_isolation") == "ok"
    assert _status(report, "fit_trace_model_content") == "ok"
    assert _status(report, "fit_trace_preprocessor_isolation") == "not_applicable"


def test_declared_no_preprocessing_conflicts_with_recorded_fit() -> None:
    x, test_x, _, _, tr_ids, te_ids = _split()
    rec = FitRecorder()
    rec.fit(StandardScaler(), x, row_ids=tr_ids, role="preprocessor")
    audit = audit_fit_trace(rec.records, train_features=x, evaluation_features=test_x,
                            train_ids=tr_ids, evaluation_ids=te_ids, preprocessor_applied=False)
    lookup = {item.code: item.status for item in audit}
    assert lookup["fit_trace_preprocessor_isolation"] == "problem"


def test_source_buffers_do_not_change_during_observation() -> None:
    x, test_x, y, _, tr_ids, te_ids = _split()
    original_x = x.copy()
    original_eval = test_x.copy()
    rec = FitRecorder()
    rec.fit(LogisticRegression(max_iter=700), x, y, row_ids=tr_ids, role="model")
    audit_fit_trace(rec.records, train_features=x, evaluation_features=test_x,
                    train_ids=tr_ids, evaluation_ids=te_ids)
    np.testing.assert_array_equal(x, original_x)
    np.testing.assert_array_equal(test_x, original_eval)
