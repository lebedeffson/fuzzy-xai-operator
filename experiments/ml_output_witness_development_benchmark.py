"""Developmental output-call comparison on four actual sklearn pipelines.

Witnesses come from caller-initiated predict/predict_proba calls. Ordinary
checks and an equally privileged direct comparator are explicitly included.
This is a controlled experiment, not external blinded fault detection.
"""

from __future__ import annotations

import hashlib
import json
import platform
from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np
import sklearn
from fuzzyxai.ml_observer import MLRunEvidence, observe_ml_run
from fuzzyxai.output_provenance import OutputRecorder
from sklearn.cluster import KMeans
from sklearn.datasets import load_diabetes, load_iris, load_wine
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.utils.validation import check_array

PROTOCOL_PATH = Path(__file__).with_name("ml_output_witness_development_protocol_v1.json")


def _make_model(name: str, seed: int) -> Any:
    if name == "DecisionTreeClassifier":
        return DecisionTreeClassifier(random_state=seed)
    if name == "RandomForestClassifier":
        return RandomForestClassifier(n_estimators=40, random_state=seed)
    if name == "LinearRegression":
        return LinearRegression()
    if name == "KMeans":
        return KMeans(n_clusters=3, n_init=10, random_state=seed)
    raise ValueError("Unrecognized protocol model")


def _dataset(name: str, task: str, seed: int) -> tuple[np.ndarray, np.ndarray, np.ndarray | None, np.ndarray | None, tuple[str, ...]]:
    dataset = {
        "iris": load_iris,
        "wine": load_wine,
        "diabetes": load_diabetes,
        "iris_clustering": load_iris,
    }[name]()
    features = np.asarray(dataset.data, dtype=float)
    labels = None if task == "clustering" else np.asarray(dataset.target)
    train, evaluation = train_test_split(
        np.arange(len(features)), test_size=0.25, random_state=seed,
        stratify=labels if task == "classification" else None,
    )
    return (
        features[train].copy(),
        features[evaluation].copy(),
        labels[train].copy() if labels is not None else None,
        labels[evaluation].copy() if labels is not None else None,
        tuple(f"{name}:{int(i)}" for i in evaluation),
    )


def _baseline(evidence: MLRunEvidence) -> bool:
    """Ordinary numeric schema, finite, ID uniqueness, probability geometry."""
    try:
        X = check_array(evidence.evaluation_features, dtype=float)
        native_prediction = np.asarray(evidence.predictions)
        if native_prediction.dtype.kind not in "biuf":
            return True
        predicted = np.asarray(native_prediction, dtype=float)
        if predicted.ndim != 1 or len(predicted) != len(X) or not np.isfinite(predicted).all():
            return True
        if evidence.evaluation_ids is not None and (
            len(evidence.evaluation_ids) != len(X)
            or len(set(evidence.evaluation_ids)) != len(X)
        ):
            return True
        if evidence.task == "classification" and evidence.probabilities is not None:
            p = np.asarray(evidence.probabilities, dtype=float)
            if (
                p.ndim != 2 or len(p) != len(X) or p.shape[1] < 2
                or not np.isfinite(p).all()
                or not np.all((p >= 0) & (p <= 1))
                or not np.allclose(p.sum(axis=1), 1.0, atol=1e-5, rtol=0)
            ):
                return True
            classes = evidence.probability_classes
            if classes is not None:
                if len(classes) != p.shape[1] or len(set(classes)) != len(classes):
                    return True
                numeric_classes = np.asarray(classes, dtype=float)
                if not np.isfinite(numeric_classes).all():
                    return True
                from_probabilities = numeric_classes[p.argmax(axis=1)]
                if not np.array_equal(from_probabilities, predicted):
                    return True
    except (TypeError, ValueError):
        return True
    return False


def _oracle_mismatch(clean: MLRunEvidence, changed: MLRunEvidence) -> bool:
    """Equal-trust comparison: direct access to real call X, IDs and outputs."""
    return (
        not np.array_equal(clean.evaluation_features, changed.evaluation_features)
        or clean.evaluation_ids != changed.evaluation_ids
        or not np.array_equal(clean.predictions, changed.predictions)
        or (
            clean.probabilities is not None
            and not np.array_equal(clean.probabilities, changed.probabilities)
        )
        or clean.probability_classes != changed.probability_classes
    )


def _check_status(report: Any, code: str) -> str:
    found = [item.status for item in report.checks if item.code == code]
    if len(found) != 1:
        raise AssertionError(f"Expected one {code} check, found {found}")
    return found[0]


def _scenarios(clean: MLRunEvidence) -> dict[str, MLRunEvidence]:
    result: dict[str, MLRunEvidence] = {"clean": clean}
    y = np.asarray(clean.predictions).copy()
    candidates = np.flatnonzero(y != y[0])
    if not len(candidates):
        raise ValueError("No different observed predictions to create a meaningful row swap")
    index = int(candidates[0])
    y[0], y[index] = y[index], y[0]
    result["reordered_prediction_values"] = replace(clean, predictions=y)

    features = np.asarray(clean.evaluation_features).copy()
    features[[0, 1]] = features[[1, 0]]
    result["reordered_report_features"] = replace(clean, evaluation_features=features)

    reordered = list(clean.evaluation_ids)
    reordered[0], reordered[1] = reordered[1], reordered[0]
    result["reordered_report_ids"] = replace(clean, evaluation_ids=tuple(reordered))

    result["missing_output_record"] = replace(clean, output_records=None)

    if clean.task == "classification":
        probabilities = np.asarray(clean.probabilities).copy()
        probabilities[:, [0, 1]] = probabilities[:, [1, 0]]
        classes = list(clean.probability_classes)
        classes[0], classes[1] = classes[1], classes[0]
        result["permuted_probability_columns_with_consistent_class_map"] = replace(
            clean, probabilities=probabilities, probability_classes=tuple(classes)
        )
    return result


def run_benchmark() -> dict[str, Any]:
    protocol_raw = PROTOCOL_PATH.read_bytes()
    protocol = json.loads(protocol_raw)
    seed = int(protocol["seed"])
    records: list[dict[str, Any]] = []
    for data in protocol["datasets"]:
        name, task, estimator_name = data["name"], data["task"], data["model"]
        X_train, X_eval, y_train, y_eval, eval_ids = _dataset(name, task, seed)
        model = _make_model(estimator_name, seed)
        if y_train is None:
            model.fit(X_train)
        else:
            model.fit(X_train, y_train)
        if type(model).__name__ != estimator_name:
            raise AssertionError("Benchmark model differs from fixed protocol")

        recorder = OutputRecorder()
        predicted = recorder.predict(model, X_eval, row_ids=eval_ids)
        probabilities = (
            recorder.predict_proba(model, X_eval, row_ids=eval_ids)
            if task == "classification" else None
        )
        clean = MLRunEvidence(
            task=task, model_family=estimator_name,
            train_features=X_train, evaluation_features=X_eval,
            train_targets=y_train, evaluation_targets=y_eval,
            predictions=predicted, probabilities=probabilities,
            probability_classes=tuple(str(v) for v in model.classes_) if probabilities is not None else None,
            evaluation_ids=eval_ids,
            output_records=tuple(entry.to_dict() for entry in recorder.records),
        )
        for case, evidence in _scenarios(clean).items():
            if case not in {scenario["name"] for scenario in protocol["scenarios"]}:
                raise AssertionError("Benchmark created an undeclared case")
            observation = observe_ml_run(evidence)
            codes = [
                "output_trace_predict_row_ids",
                "output_trace_predict_inputs",
                "output_trace_predict_values",
                "output_trace_predict_proba_values",
                "output_trace_predict_proba_classes",
            ]
            statuses = {}
            for code in codes:
                found = [entry.status for entry in observation.checks if entry.code == code]
                statuses[code] = found[0] if len(found) == 1 else None
            missing = [entry.status for entry in observation.checks if entry.code == "output_trace"]
            records.append({
                "dataset": name, "task": task, "estimator": type(model).__name__,
                "case": case,
                "basic_numeric_alert": _baseline(evidence),
                "privileged_direct_comparison_alert": _oracle_mismatch(clean, evidence),
                "output_trace_missing_status": missing[0] if len(missing) == 1 else None,
                "checks": statuses,
            })
    totals = {}
    for scenario in protocol["scenarios"]:
        relevant = [entry for entry in records if entry["case"] == scenario["name"]]
        totals[scenario["name"]] = {
            "n": len(relevant),
            "basic_numeric_alerts": sum(v["basic_numeric_alert"] for v in relevant),
            "privileged_direct_alerts": sum(v["privileged_direct_comparison_alert"] for v in relevant),
            "observed_problem": sum(
                "problem" in v["checks"].values() for v in relevant
            ),
            "missing_unknown": sum(v["output_trace_missing_status"] == "unknown" for v in relevant),
        }
    return {
        "phase": protocol["phase"],
        "protocol_id": protocol["id"],
        "protocol_sha256": hashlib.sha256(protocol_raw).hexdigest(),
        "environment": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "scikit_learn": sklearn.__version__,
        },
        "n_datasets": len(protocol["datasets"]),
        "n_cases": len(records),
        "by_scenario": totals,
        "records": records,
        "limits": protocol["guardrails"],
    }


if __name__ == "__main__":
    print(json.dumps(run_benchmark(), indent=2, ensure_ascii=False, sort_keys=True, allow_nan=False))
