"""H5a action-contract validation and separately scoped H5b prediction analysis."""

from __future__ import annotations

from typing import Sequence

import numpy as np

from .contracts import ConfirmatoryProtocolLock, PredictiveRouteExample, ResearchPartition, RouteContractRecord


def evaluate_action_contract(records: Sequence[RouteContractRecord]) -> dict[str, object]:
    """Measure whether invalid routes are prevented from automatic certification."""
    if not records:
        raise ValueError("route-contract evaluation requires records")
    truth = np.asarray([record.contract_violated for record in records], dtype=bool)
    predicted = np.asarray([record.requested_action.value != "accept" for record in records], dtype=bool)
    true_positive = int(np.sum(truth & predicted))
    false_positive = int(np.sum(~truth & predicted))
    false_negative = int(np.sum(truth & ~predicted))
    precision = true_positive / max(1, true_positive + false_positive)
    recall = true_positive / max(1, true_positive + false_negative)
    f1 = 2.0 * precision * recall / max(1e-12, precision + recall)
    invalid_accepts = sum(record.invalid_automatic_action for record in records)
    return {
        "target_event": "automatic action allowed while the mandatory explanation contract is violated",
        "n_records": len(records),
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "false_certification_rate": invalid_accepts / max(1, int(truth.sum())),
        "false_refusal_rate": false_positive / max(1, int((~truth).sum())),
        "fault_type_accuracy": _fault_field_accuracy(records, "fault_type"),
        "fault_source_accuracy": _fault_field_accuracy(records, "fault_source"),
        "structural_claim_allowed": f1 >= 0.95 and invalid_accepts / max(1, int(truth.sum())) <= 0.01,
        "model_error_prediction_claim_allowed": False,
    }


def evaluate_predictive_increment(
    development: Sequence[PredictiveRouteExample],
    test: Sequence[PredictiveRouteExample],
    protocol_lock: ConfirmatoryProtocolLock,
    *,
    bootstrap_repetitions: int = 1000,
    seed: int = 4201,
) -> dict[str, object]:
    """Compare M0 and M1 on an untouched test partition without conflating H5a."""
    if not development or not test:
        raise ValueError("predictive comparison requires development and test records")
    if any(item.partition is ResearchPartition.TEST for item in development):
        raise ValueError("development data contains confirmatory test records")
    if any(item.partition is not ResearchPartition.TEST for item in test):
        raise ValueError("test records must use the test partition")
    if any(not item.source_features_are_oof for item in development):
        raise ValueError("development prediction features must be out-of-fold")

    baseline_train = np.asarray([item.baseline_features for item in development], dtype=float)
    typed_train = np.asarray([(*item.baseline_features, *item.typed_route_features) for item in development], dtype=float)
    labels_train = np.asarray([item.model_error for item in development], dtype=int)
    baseline_test = np.asarray([item.baseline_features for item in test], dtype=float)
    typed_test = np.asarray([(*item.baseline_features, *item.typed_route_features) for item in test], dtype=float)
    labels_test = np.asarray([item.model_error for item in test], dtype=int)
    if len(np.unique(labels_train)) != 2 or len(np.unique(labels_test)) != 2:
        raise ValueError("both partitions must contain errors and correct predictions")

    scores_m0 = _fit_scores(baseline_train, labels_train, baseline_test, seed)
    scores_m1 = _fit_scores(typed_train, labels_train, typed_test, seed)
    from sklearn.metrics import average_precision_score

    m0 = float(average_precision_score(labels_test, scores_m0))
    m1 = float(average_precision_score(labels_test, scores_m1))
    interval = _paired_bootstrap(labels_test, scores_m0, scores_m1, bootstrap_repetitions, seed)
    supported = interval[0] > 0.0
    return {
        "target": "held-out model-error association",
        "protocol_sha256": protocol_lock.protocol_sha256,
        "m0_auprc": m0,
        "m1_auprc": m1,
        "incremental_auprc": m1 - m0,
        "confidence_interval_95": interval,
        "status": "supported" if supported else "not_supported",
        "predictive_claim_allowed": supported,
        "structural_h5a_unchanged": True,
        "test_opened_once": True,
    }


def _fit_scores(train: np.ndarray, labels: np.ndarray, test: np.ndarray, seed: int) -> np.ndarray:
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    model = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000, class_weight="balanced", random_state=seed))
    model.fit(train, labels)
    result: np.ndarray = np.asarray(model.predict_proba(test)[:, 1], dtype=float)
    return result


def _paired_bootstrap(
    labels: np.ndarray,
    scores_m0: np.ndarray,
    scores_m1: np.ndarray,
    repetitions: int,
    seed: int,
) -> list[float]:
    from sklearn.metrics import average_precision_score

    rng = np.random.default_rng(seed)
    differences = []
    for _ in range(repetitions):
        sample = rng.integers(0, len(labels), size=len(labels))
        sampled_labels = labels[sample]
        if len(np.unique(sampled_labels)) < 2:
            continue
        differences.append(
            float(average_precision_score(sampled_labels, scores_m1[sample])) - float(average_precision_score(sampled_labels, scores_m0[sample]))
        )
    if not differences:
        return [0.0, 0.0]
    return [float(np.quantile(differences, 0.025)), float(np.quantile(differences, 0.975))]


def _fault_field_accuracy(records: Sequence[RouteContractRecord], field: str) -> float:
    faulted = [record for record in records if record.contract_violated]
    if not faulted:
        return 1.0
    detected_field = f"detected_{field}"
    matches = sum(int(getattr(record, field) == getattr(record, detected_field)) for record in faulted)
    return float(matches / len(faulted))
