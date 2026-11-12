"""End-to-end data -> explicit fit -> explicit predictions -> read-only audit."""

from __future__ import annotations

from dataclasses import asdict, replace

import numpy as np
from fuzzyxai.fit_provenance import FitRecorder
from fuzzyxai.ml_observer import MLRunEvidence, observe_ml_run
from fuzzyxai.output_provenance import OutputRecorder
from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier

KEY = b"test-only-target-secret-do-not-deploy"
CHECKS = (
    "fit_trace_model_isolation", "fit_trace_model_content",
    "fit_trace_model_targets", "output_trace_predict_row_ids",
    "output_trace_predict_inputs", "output_trace_predict_values",
    "output_trace_predict_proba_row_ids", "output_trace_predict_proba_inputs",
    "output_trace_predict_proba_values", "output_trace_predict_proba_classes",
)


def make_run() -> MLRunEvidence:
    data = load_wine()
    train, test = train_test_split(
        np.arange(len(data.target)), random_state=58, test_size=0.25,
        stratify=data.target,
    )
    X_train, X_test = data.data[train].copy(), data.data[test].copy()
    y_train, y_test = data.target[train].copy(), data.target[test].copy()
    tr_ids = tuple(f"wine:{int(i)}" for i in train)
    ev_ids = tuple(f"wine:{int(i)}" for i in test)
    recorder = FitRecorder(target_hmac_key=KEY)
    model = recorder.fit(
        DecisionTreeClassifier(random_state=58), X_train, y_train,
        role="model", row_ids=tr_ids,
    )
    outputs = OutputRecorder()
    pred = outputs.predict(model, X_test, row_ids=ev_ids)
    proba = outputs.predict_proba(model, X_test, row_ids=ev_ids)
    names = tuple(str(x) for x in data.feature_names)
    return MLRunEvidence(
        task="classification", model_family=type(model).__name__,
        train_features=X_train, evaluation_features=X_test,
        train_targets=y_train, evaluation_targets=y_test,
        train_feature_names=names, evaluation_feature_names=names,
        predictions=pred, probabilities=proba,
        probability_classes=tuple(str(x) for x in model.classes_),
        train_ids=tr_ids, evaluation_ids=ev_ids,
        model_fit_ids=tr_ids, preprocessor_applied=False,
        model_version="wine-tree-58", random_seed=58,
        fit_records=[record.to_dict() for record in recorder.records],
        output_records=[record.to_dict() for record in outputs.records],
    )


def statuses(evidence: MLRunEvidence, *, key: bytes | None = KEY) -> dict[str, str]:
    return {
        item.code: item.status
        for item in observe_ml_run(evidence, target_hmac_key=key).checks
    }


def test_complete_model_audit_passes_specific_witnesses_without_mutation() -> None:
    evidence = make_run()
    training_copy = np.asarray(evidence.train_features).copy()
    evaluation_copy = np.asarray(evidence.evaluation_features).copy()
    original_records = [dict(x) for x in evidence.output_records]
    found = statuses(evidence)
    assert all(found[code] == "ok" for code in CHECKS)
    assert found["preprocessor_fit_isolation"] == "not_applicable"
    np.testing.assert_array_equal(evidence.train_features, training_copy)
    np.testing.assert_array_equal(evidence.evaluation_features, evaluation_copy)
    assert evidence.output_records == original_records


def test_source_target_and_output_corruption_are_separate_checks() -> None:
    evidence = make_run()
    changed_y = replace(evidence, train_targets=np.roll(np.asarray(evidence.train_targets), 1))
    targets = statuses(changed_y)
    assert targets["fit_trace_model_targets"] == "problem"
    assert targets["fit_trace_model_content"] == "ok"
    assert targets["output_trace_predict_values"] == "ok"

    wrong = np.asarray(evidence.predictions).copy()
    offset = int(np.flatnonzero(wrong != wrong[0])[0])
    wrong[0], wrong[offset] = wrong[offset], wrong[0]
    output = statuses(replace(evidence, predictions=wrong))
    assert output["fit_trace_model_targets"] == "ok"
    assert output["output_trace_predict_values"] == "problem"


def test_missing_source_secret_is_unknown_but_output_is_still_observed() -> None:
    evidence = make_run()
    found = statuses(evidence, key=None)
    assert found["fit_trace_model_targets"] == "unknown"
    assert found["fit_trace_model_content"] == "ok"
    assert found["output_trace_predict_values"] == "ok"


def test_serialized_evidence_reopens_without_model_or_inference() -> None:
    state = asdict(make_run())
    for name in (
        "train_features", "evaluation_features", "train_targets",
        "evaluation_targets", "predictions", "probabilities",
    ):
        state[name] = np.asarray(state[name]).tolist()
    recreated = MLRunEvidence(**state)
    found = statuses(recreated)
    assert all(found[code] == "ok" for code in CHECKS)
