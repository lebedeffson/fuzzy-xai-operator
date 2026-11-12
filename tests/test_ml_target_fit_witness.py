"""Independent target-value checks for explicit, caller-initiated fit witnesses.

These are source-consistency tests, not tests that ground-truth labels are
correct. A private HMAC key stays outside serialized JSON evidence.
"""

from __future__ import annotations

import json
from dataclasses import asdict

import numpy as np
import pytest
from fuzzyxai.fit_provenance import FitRecorder, audit_fit_trace
from fuzzyxai.ml_observer import MLRunEvidence, observe_ml_run
from sklearn.datasets import load_digits, load_linnerud, load_wine
from sklearn.linear_model import Ridge
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier

TEST_HMAC_KEY = b"reproducible-test-key-must-not-use-in-prod"


def _dataset(name: str):
    data = {"wine": load_wine, "digits": load_digits, "linnerud": load_linnerud}[name]()
    x = np.asarray(data.data, dtype=float)
    y = np.asarray(data.target, dtype=float)
    train, evaluation = train_test_split(
        np.arange(len(x)),
        random_state=917,
        test_size=0.25,
        stratify=y if name != "linnerud" else None,
    )
    train_ids = tuple(f"{name}:{i}" for i in train)
    eval_ids = tuple(f"{name}:{i}" for i in evaluation)
    return x[train].copy(), x[evaluation].copy(), y[train].copy(), y[evaluation].copy(), train_ids, eval_ids


def _source(name: str, *, permute_targets: bool = False):
    x, ev, targets, ev_targets, train_ids, eval_ids = _dataset(name)
    recorder = FitRecorder(target_hmac_key=TEST_HMAC_KEY)
    used_targets = np.roll(targets, 1, axis=0) if permute_targets else targets
    model = Ridge() if name == "linnerud" else DecisionTreeClassifier(random_state=917)
    recorder.fit(model, x, used_targets, role="model", row_ids=train_ids)
    evidence = MLRunEvidence(
        task="regression" if name == "linnerud" else "classification",
        model_family=type(model).__name__,
        train_features=x,
        evaluation_features=ev,
        train_targets=targets,
        evaluation_targets=ev_targets,
        train_ids=train_ids,
        evaluation_ids=eval_ids,
        model_fit_ids=train_ids,
        preprocessor_applied=False,
        fit_records=[record.to_dict() for record in recorder.records],
    )
    return recorder, evidence


def _code(report, wanted: str) -> str:
    checks = [x.status for x in report.checks if x.code == wanted]
    assert len(checks) == 1, (wanted, checks)
    return checks[0]


@pytest.mark.parametrize("name", ["wine", "digits", "linnerud"])
def test_real_new_dataset_fit_targets_match_reference_and_witness_is_read_only(name: str) -> None:
    recorder, evidence = _source(name)
    before = np.asarray(evidence.train_targets).copy()
    outcome = observe_ml_run(evidence, target_hmac_key=TEST_HMAC_KEY)
    assert _code(outcome, "fit_trace_model_isolation") == "ok"
    assert _code(outcome, "fit_trace_model_content") == "ok"
    assert _code(outcome, "fit_trace_model_targets") == "ok"
    np.testing.assert_array_equal(evidence.train_targets, before)
    saved = json.dumps(recorder.records[0].to_dict())
    assert "target_batch_hmac" in saved
    assert "target_status" in saved
    assert TEST_HMAC_KEY.decode() not in saved
    assert "target_rows" not in saved
    assert "labels" not in saved


@pytest.mark.parametrize("name", ["wine", "digits", "linnerud"])
def test_target_permutation_preserves_feature_witness_but_breaks_label_witness(name: str) -> None:
    recorder, evidence = _source(name, permute_targets=True)
    report = observe_ml_run(evidence, target_hmac_key=TEST_HMAC_KEY)
    assert _code(report, "fit_trace_model_content") == "ok"
    assert _code(report, "fit_trace_model_isolation") == "ok"
    assert _code(report, "fit_trace_model_targets") == "problem"
    assert "fit_trace_model_targets" in {f.code for f in report.checks if f.status == "problem"}
    assert recorder.records[0].target_status == "witnessed"


@pytest.mark.parametrize("name", ["wine", "digits"])
def test_no_secret_does_not_implicitly_certify_target_values(name: str) -> None:
    _, evidence = _source(name)
    report = observe_ml_run(evidence)
    assert _code(report, "fit_trace_model_targets") == "unknown"
    assert _code(report, "fit_trace_model_content") == "ok"


def test_legacy_records_without_target_witness_remain_backward_compatible() -> None:
    rec, evidence = _source("wine")
    legacy = rec.records[0].to_dict()
    del legacy["target_status"]
    del legacy["target_batch_hmac"]
    legacy_evidence = MLRunEvidence(**{**asdict(evidence), "fit_records": [legacy]})
    report = observe_ml_run(legacy_evidence, target_hmac_key=TEST_HMAC_KEY)
    assert _code(report, "fit_trace_model_targets") == "unknown"
    assert _code(report, "fit_trace_model_content") == "ok"


def test_corrupted_label_hmac_is_rejected_by_trace_validator() -> None:
    _, evidence = _source("wine")
    altered = evidence.fit_records[0].copy()
    altered["target_batch_hmac"] = "x" * 64
    report = observe_ml_run(
        MLRunEvidence(**{**asdict(evidence), "fit_records": [altered]}),
        target_hmac_key=TEST_HMAC_KEY,
    )
    assert _code(report, "fit_trace") == "problem"


def test_missing_reference_training_labels_are_unknown_not_a_false_pass() -> None:
    _, evidence = _source("wine")
    no_reference = MLRunEvidence(**{**asdict(evidence), "train_targets": None})
    report = observe_ml_run(no_reference, target_hmac_key=TEST_HMAC_KEY)
    assert _code(report, "fit_trace_model_targets") == "unknown"


def test_wrong_target_reference_is_not_diagnosed_as_true_or_false_labels() -> None:
    _, evidence = _source("wine")
    reference = np.asarray(evidence.train_targets)
    declared = MLRunEvidence(**{**asdict(evidence), "train_targets": np.roll(reference, 2)})
    report = observe_ml_run(declared, target_hmac_key=TEST_HMAC_KEY)
    assert _code(report, "fit_trace_model_targets") == "problem"
    # A mismatch cannot establish whether the witness or reference is correct.


def test_no_target_fit_is_not_applicable() -> None:
    x, ev, _, _, train_ids, eval_ids = _dataset("wine")
    recorder = FitRecorder(target_hmac_key=TEST_HMAC_KEY)

    class UnsupervisedEstimator:
        def fit(self, X):
            return self

    recorder.fit(UnsupervisedEstimator(), x, row_ids=train_ids, role="model")
    result = audit_fit_trace(
        recorder.records,
        train_features=x, evaluation_features=ev,
        train_ids=train_ids, evaluation_ids=eval_ids,
        target_hmac_key=TEST_HMAC_KEY,
    )
    assert {item.code: item.status for item in result}["fit_trace_model_targets"] == "not_applicable"


def test_non_numeric_class_labels_are_unknown_and_remain_valid_model_inputs() -> None:
    x, ev, targets, _, train_ids, eval_ids = _dataset("wine")
    text_labels = np.asarray([f"category_{int(v)}" for v in targets])
    recorder = FitRecorder(target_hmac_key=TEST_HMAC_KEY)
    fitted = recorder.fit(
        DecisionTreeClassifier(random_state=917), x, text_labels,
        role="model", row_ids=train_ids,
    )
    assert fitted.classes_.size >= 2
    assert recorder.records[0].target_status == "unavailable"
    result = audit_fit_trace(
        recorder.records, train_features=x, evaluation_features=ev,
        train_ids=train_ids, evaluation_ids=eval_ids, train_targets=text_labels,
        target_hmac_key=TEST_HMAC_KEY,
    )
    assert {check.code: check.status for check in result}["fit_trace_model_targets"] == "unknown"


@pytest.mark.parametrize("secret", [b"short", "", 123])
def test_unusable_secret_cannot_be_mistaken_for_secure_target_commitment(secret) -> None:
    with pytest.raises(ValueError, match="target_hmac_key"):
        FitRecorder(target_hmac_key=secret)


def test_json_cli_cannot_verify_label_hmac_without_a_key() -> None:
    _, evidence = _source("wine")
    roundtrip = json.loads(json.dumps(asdict(evidence), default=lambda x: x.tolist()))
    report = observe_ml_run(MLRunEvidence(**roundtrip))
    assert _code(report, "fit_trace_model_targets") == "unknown"



def test_target_hmac_cannot_be_replayed_onto_different_feature_batch() -> None:
    x, eval_x, train_y, eval_y, train_ids, eval_ids = _dataset("wine")
    original = FitRecorder(target_hmac_key=TEST_HMAC_KEY)
    changed = FitRecorder(target_hmac_key=TEST_HMAC_KEY)
    original.fit(DecisionTreeClassifier(random_state=917), x, train_y, role="model", row_ids=train_ids)
    shifted = x.copy()
    shifted[:, 0] = shifted[:, 0] + 0.125
    changed.fit(DecisionTreeClassifier(random_state=917), shifted, train_y, role="model", row_ids=train_ids)
    assert original.records[0].batch_sha256 != changed.records[0].batch_sha256
    assert original.records[0].target_batch_hmac != changed.records[0].target_batch_hmac

    swapped_evidence = changed.records[0].to_dict()
    swapped_evidence["target_batch_hmac"] = original.records[0].target_batch_hmac
    report = observe_ml_run(
        MLRunEvidence(
            task="classification",
            model_family="DecisionTreeClassifier",
            train_features=shifted,
            evaluation_features=eval_x,
            train_targets=train_y,
            evaluation_targets=eval_y,
            train_ids=train_ids,
            evaluation_ids=eval_ids,
            model_fit_ids=train_ids,
            preprocessor_applied=False,
            fit_records=[swapped_evidence],
        ),
        target_hmac_key=TEST_HMAC_KEY,
    )
    assert _code(report, "fit_trace_model_content") == "ok"
    assert _code(report, "fit_trace_model_targets") == "problem"



def test_numeric_string_labels_are_not_coerced_into_numeric_target_certificate() -> None:
    x, ev, y, _, train_ids, eval_ids = _dataset("wine")
    text_values = np.asarray([str(int(v)) for v in y], dtype=str)
    recorder = FitRecorder(target_hmac_key=TEST_HMAC_KEY)
    recorder.fit(DecisionTreeClassifier(random_state=917), x, text_values, role="model", row_ids=train_ids)
    assert recorder.records[0].target_status == "unavailable"
    observations = audit_fit_trace(
        recorder.records,
        train_features=x, evaluation_features=ev,
        train_ids=train_ids, evaluation_ids=eval_ids,
        train_targets=y,
        target_hmac_key=TEST_HMAC_KEY,
    )
    assert {entry.code: entry.status for entry in observations}["fit_trace_model_targets"] == "unknown"


def test_large_integer_labels_cannot_collapse_under_float64_canonicalization() -> None:
    class PermissiveEstimator:
        def fit(self, X, y):
            return self

    x, ev, _, _, train_ids, eval_ids = _dataset("wine")
    unusual = np.full(len(x), 2**53 + 1, dtype=np.int64)
    rec = FitRecorder(target_hmac_key=TEST_HMAC_KEY)
    rec.fit(PermissiveEstimator(), x, unusual, row_ids=train_ids, role="model")
    assert rec.records[0].target_status == "unavailable"
    result = audit_fit_trace(
        rec.records, train_features=x, evaluation_features=ev,
        train_ids=train_ids, evaluation_ids=eval_ids,
        train_targets=unusual, target_hmac_key=TEST_HMAC_KEY,
    )
    assert {item.code: item.status for item in result}["fit_trace_model_targets"] == "unknown"



def test_valid_uint64_numeric_targets_remain_supported() -> None:
    class PermissiveEstimator:
        def fit(self, X, y):
            return self

    x, ev, source, _, train_ids, eval_ids = _dataset("wine")
    values = source.astype(np.uint64)
    recorder = FitRecorder(target_hmac_key=TEST_HMAC_KEY)
    recorder.fit(PermissiveEstimator(), x, values, row_ids=train_ids, role="model")
    assert recorder.records[0].target_status == "witnessed"
    result = audit_fit_trace(
        recorder.records, train_features=x, evaluation_features=ev,
        train_ids=train_ids, evaluation_ids=eval_ids,
        train_targets=values, target_hmac_key=TEST_HMAC_KEY,
    )
    assert {entry.code: entry.status for entry in result}["fit_trace_model_targets"] == "ok"
