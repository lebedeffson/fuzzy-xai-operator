"""Developer-constructed real sklearn fit-call provenance comparison.

Runs the locked-in-development protocol in ml_fit_source_witness_protocol_v1.json.
The observer only consumes trace evidence; intentionally broken models are
trained by this experiment outside the observer itself. No test accuracy of
models exposed to evaluation data is interpreted as generalization.
"""

from __future__ import annotations

import hashlib
import json
import platform
from pathlib import Path
from typing import Any

import numpy as np
import sklearn
from fuzzyxai.fit_provenance import FitRecorder
from fuzzyxai.ml_observer import MLRunEvidence, observe_ml_run
from sklearn.cluster import KMeans
from sklearn.datasets import load_breast_cancer, load_diabetes, load_iris
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.model_selection import train_test_split

PROTOCOL_PATH = Path(__file__).with_name("ml_fit_source_witness_protocol_v1.json")


def _load_data(name: str, task: str, seed: int) -> tuple[np.ndarray, np.ndarray, np.ndarray | None, np.ndarray | None, tuple[str, ...], tuple[str, ...]]:
    dataset = {
        "iris": load_iris,
        "breast_cancer": load_breast_cancer,
        "diabetes": load_diabetes,
        "iris_clustering": load_iris,
    }[name]()
    values = np.asarray(dataset.data, dtype=float)
    target = None if task == "clustering" else np.asarray(dataset.target)
    train, evaluation = train_test_split(
        np.arange(len(values)),
        random_state=seed,
        test_size=0.25,
        stratify=target if task == "classification" else None,
    )
    return (
        values[train].copy(),
        values[evaluation].copy(),
        target[train] if target is not None else None,
        target[evaluation] if target is not None else None,
        tuple(f"{name}-{index}" for index in train),
        tuple(f"{name}-{index}" for index in evaluation),
    )


def _estimator(estimator_name: str, seed: int) -> Any:
    """Use the literal estimator type fixed in the committed protocol."""
    if estimator_name == "LogisticRegression":
        return LogisticRegression(max_iter=800, solver="newton-cholesky", random_state=seed)
    if estimator_name == "LinearRegression":
        return LinearRegression()
    if estimator_name == "KMeans":
        return KMeans(n_clusters=3, random_state=seed, n_init=10)
    raise ValueError(f"Unsupported protocol estimator: {estimator_name}")


def _status(report: Any, code: str) -> str:
    result = [check.status for check in report.checks if check.code == code]
    if len(result) != 1:
        raise AssertionError(f"not exactly one {code} result: {result}")
    return result[0]


def run_benchmark() -> dict[str, Any]:
    protocol_raw = PROTOCOL_PATH.read_bytes()
    protocol = json.loads(protocol_raw)
    rows: list[dict[str, Any]] = []
    seed = int(protocol["seed"])

    for descriptor in protocol["datasets"]:
        name = descriptor["name"]
        task = descriptor["task"]
        estimator_name = descriptor["estimator"]
        x_train, x_eval, y_train, y_eval, train_ids, eval_ids = _load_data(name, task, seed)
        for scenario in protocol["scenarios"]:
            name_of_case = scenario["name"]
            recorder = FitRecorder()
            declared_fit_ids: tuple[str, ...]
            fitted = _estimator(estimator_name, seed)

            if name_of_case == "clean":
                fitted = recorder.fit(
                    fitted, x_train, y_train, row_ids=train_ids, role="model",
                )
                declared_fit_ids = train_ids
            elif name_of_case == "fit_evaluation_truthful_ids":
                both_x = np.vstack([x_train, x_eval])
                both_y = np.concatenate([y_train, y_eval]) if y_train is not None and y_eval is not None else None
                fitted = recorder.fit(
                    fitted, both_x, both_y,
                    row_ids=train_ids + eval_ids, role="model",
                )
                declared_fit_ids = train_ids + eval_ids
            elif name_of_case == "fit_evaluation_false_ids":
                # Misleading input IDs are derived solely from training rows.
                # Recorded feature fingerprints still reflect evaluation data.
                fitted = recorder.fit(
                    fitted, x_eval, y_eval,
                    row_ids=train_ids[:len(x_eval)], role="model",
                )
                declared_fit_ids = train_ids[:len(x_eval)]
            elif name_of_case == "fit_evaluation_true_trace_false_metadata":
                fitted = recorder.fit(
                    fitted, x_eval, y_eval,
                    row_ids=eval_ids, role="model",
                )
                declared_fit_ids = train_ids[:len(x_eval)]
            else:
                # Valid model fit occurs without our instrumented wrapper.
                # The observer cannot infer which rows were actually seen.
                if y_train is None:
                    fitted.fit(x_train)
                else:
                    fitted.fit(x_train, y_train)
                declared_fit_ids = train_ids

            evidence = MLRunEvidence(
                task=task,
                model_family=type(fitted).__name__,
                train_features=x_train,
                evaluation_features=x_eval,
                train_targets=y_train,
                evaluation_targets=None,
                predictions=None,  # Avoid contaminated-model evaluation metrics.
                train_ids=train_ids,
                evaluation_ids=eval_ids,
                model_fit_ids=declared_fit_ids,
                preprocessor_applied=False,
                fit_records=tuple(record.to_dict() for record in recorder.records) if recorder.records else None,
                model_version=f"{name}-{seed}",
                random_seed=seed,
            )
            report = observe_ml_run(evidence)
            metadata_detected = bool(set(declared_fit_ids) & set(eval_ids))
            trace_isolation = _status(report, "fit_trace_model_isolation") if recorder.records else "unknown"
            trace_content = _status(report, "fit_trace_model_content") if recorder.records else "unknown"
            detected = trace_isolation == "problem" or trace_content == "problem"
            rows.append(
                {
                    "dataset": name,
                    "task": task,
                    "estimator": type(fitted).__name__,
                    "protocol_estimator": estimator_name,
                    "solver": getattr(fitted, "solver", None),
                    "training_converged": bool(np.all(fitted.n_iter_ < fitted.max_iter)) if hasattr(fitted, "n_iter_") else None,
                    "scenario": name_of_case,
                    "metadata_baseline_detected": metadata_detected,
                    "witness_detected": detected,
                    "fit_trace_isolation": trace_isolation,
                    "fit_trace_content": trace_content,
                    "witness_status": "unknown" if not recorder.records else "witnessed",
                    "observer_unknown_codes": [
                        finding.code for finding in report.checks if finding.status == "unknown"
                    ],
                }
            )

    aggregate: dict[str, dict[str, Any]] = {}
    for descriptor in protocol["scenarios"]:
        case = descriptor["name"]
        selected = [item for item in rows if item["scenario"] == case]
        aggregate[case] = {
            "n": len(selected),
            "metadata_detections": sum(item["metadata_baseline_detected"] for item in selected),
            "witness_detections": sum(item["witness_detected"] for item in selected),
            "unknown_witness": sum(item["witness_status"] == "unknown" for item in selected),
            "source_content_contradictions": sum(item["fit_trace_content"] == "problem" for item in selected),
        }
    return {
        "phase": protocol["phase"],
        "protocol_id": protocol["protocol_id"],
        "protocol_sha256": hashlib.sha256(protocol_raw).hexdigest(),
        "datasets": len(protocol["datasets"]),
        "cases": len(rows),
        "environment": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "scikit_learn": sklearn.__version__,
        },
        "by_scenario": aggregate,
        "records": rows,
        "limits": protocol["guardrails"],
    }


if __name__ == "__main__":
    print(json.dumps(run_benchmark(), indent=2, sort_keys=True, ensure_ascii=False, allow_nan=False))
