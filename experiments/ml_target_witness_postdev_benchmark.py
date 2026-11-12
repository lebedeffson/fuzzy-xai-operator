"""Post-development source-consistency study on three previously unused fit datasets.

The observer receives only evidence. This experiment explicitly fits models
and performs deliberate target-data substitutions. It is NOT a blind
independent study, and no contaminated model scores are reported.
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
from sklearn.datasets import load_digits, load_linnerud, load_wine
from sklearn.linear_model import Ridge
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier

PROTOCOL_FILE = Path(__file__).with_name("ml_target_witness_postdev_protocol_v1.json")
BENCHMARK_KEY = b"developmental-benchmark-hmac-secret-do-not-deploy"


def _dataset(name: str, task: str, seed: int) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, tuple[str, ...], tuple[str, ...]]:
    dataset = {"wine": load_wine, "digits": load_digits, "linnerud": load_linnerud}[name]()
    x = np.asarray(dataset.data, dtype=float)
    y = np.asarray(dataset.target, dtype=float)
    train_idx, test_idx = train_test_split(
        np.arange(len(x)),
        random_state=seed,
        test_size=0.25,
        stratify=y if task == "classification" else None,
    )
    return (
        x[train_idx].copy(),
        x[test_idx].copy(),
        y[train_idx].copy(),
        y[test_idx].copy(),
        tuple(f"{name}:{idx}" for idx in train_idx),
        tuple(f"{name}:{idx}" for idx in test_idx),
    )


def _model(estimator_name: str, seed: int) -> Any:
    if estimator_name == "DecisionTreeClassifier":
        return DecisionTreeClassifier(random_state=seed)
    if estimator_name == "Ridge":
        return Ridge()
    raise ValueError(f"Unexpected prespecified estimator: {estimator_name}")


def _check_status(report: Any, code: str) -> str:
    found = [check.status for check in report.checks if check.code == code]
    if len(found) != 1:
        raise AssertionError(f"Expected one {code}, saw {found}")
    return found[0]


def _distribution_disagrees(expected: np.ndarray, observed: np.ndarray, task: str) -> bool:
    if task == "classification":
        classes, counts = np.unique(expected, return_counts=True)
        other, other_counts = np.unique(observed, return_counts=True)
        return not (np.array_equal(classes, other) and np.array_equal(counts, other_counts))
    return not np.allclose(np.mean(expected, axis=0), np.mean(observed, axis=0), rtol=0, atol=1e-12)


def run_benchmark() -> dict[str, Any]:
    """Measure disagreement checks without modifying any source model."""

    raw = PROTOCOL_FILE.read_bytes()
    protocol = json.loads(raw)
    seed = int(protocol["seed"])
    rows: list[dict[str, Any]] = []

    for dataset_def in protocol["datasets"]:
        name = dataset_def["name"]
        task = dataset_def["task"]
        source_estimator = dataset_def["estimator"]
        x_train, x_eval, y_train, y_eval, train_ids, eval_ids = _dataset(name, task, seed)
        for scenario in protocol["scenarios"]:
            fault = scenario["name"]
            actual_targets = y_train.copy()
            declared_targets: np.ndarray | None = y_train.copy()
            protected = True

            if fault == "permuted_training_targets":
                actual_targets = np.roll(y_train, 1, axis=0)
            elif fault == "single_modified_target":
                actual_targets = y_train.copy()
                if task == "classification":
                    other = next(value for value in y_train[1:] if value != y_train[0])
                    actual_targets[0] = other
                else:
                    actual_targets[0] = actual_targets[0] + 17.0
            elif fault == "no_target_witness":
                protected = False
            elif fault == "no_target_reference":
                declared_targets = None
            elif fault == "corrupted_reference_targets":
                declared_targets = np.roll(y_train, 2, axis=0)
            elif fault != "clean":
                raise ValueError(f"Unexpected prespecified scenario: {fault}")

            recorder = FitRecorder(target_hmac_key=BENCHMARK_KEY if protected else None)
            model = _model(source_estimator, seed)
            trained = recorder.fit(
                model, x_train, actual_targets,
                role="model", row_ids=train_ids, feature_space="raw",
            )
            if type(trained).__name__ != source_estimator:
                raise AssertionError("Benchmark model does not match protocol")

            evidence = MLRunEvidence(
                task=task,
                model_family=type(trained).__name__,
                train_features=x_train,
                evaluation_features=x_eval,
                train_targets=declared_targets,
                evaluation_targets=y_eval,
                train_ids=train_ids,
                evaluation_ids=eval_ids,
                model_fit_ids=train_ids,
                preprocessor_applied=False,
                fit_records=[record.to_dict() for record in recorder.records],
                model_version=f"{name}-seed{seed}",
                random_seed=seed,
            )
            report = observe_ml_run(evidence, target_hmac_key=BENCHMARK_KEY)
            result = _check_status(report, "fit_trace_model_targets")
            content_status = _check_status(report, "fit_trace_model_content")
            ids_status = _check_status(report, "fit_trace_model_isolation")
            if content_status != "ok" or ids_status != "ok":
                raise AssertionError("The target-label experiment must not contain unrelated input/ID faults")

            # The two simpler baselines do not receive an independently
            # verified training-label event, so they cannot detect a
            # within-row-ID target substitution.
            metadata_flags = bool(set(train_ids) & set(eval_ids))
            feature_flags = content_status == "problem"
            if declared_targets is None:
                label_histogram_disagrees: bool | None = None
                direct_comparison_disagrees: bool | None = None
            else:
                # These stronger comparisons have access to the actual
                # source and fit y via the experiment oracle. Never claim
                # the HMAC audit beats an equally privileged comparator.
                label_histogram_disagrees = _distribution_disagrees(declared_targets, actual_targets, task)
                direct_comparison_disagrees = not np.array_equal(declared_targets, actual_targets)

            rows.append(
                {
                    "dataset": name,
                    "task": task,
                    "estimator": type(trained).__name__,
                    "protocol_estimator": source_estimator,
                    "fault": fault,
                    "observed_status": result,
                    "baseline_id_alert": metadata_flags,
                    "baseline_feature_alert": feature_flags,
                    "oracle_histogram_alert": label_histogram_disagrees,
                    "oracle_direct_label_alert": direct_comparison_disagrees,
                    "fit_has_target_commitment": recorder.records[0].target_status == "witnessed",
                    "label_source_available": declared_targets is not None,
                }
            )

    by_case: dict[str, dict[str, Any]] = {}
    for scenario in protocol["scenarios"]:
        fault = scenario["name"]
        selected = [row for row in rows if row["fault"] == fault]
        by_case[fault] = {
            "n": len(selected),
            "label_witness_ok": sum(row["observed_status"] == "ok" for row in selected),
            "label_witness_problem": sum(row["observed_status"] == "problem" for row in selected),
            "label_witness_unknown": sum(row["observed_status"] == "unknown" for row in selected),
            "metadata_alerts": sum(row["baseline_id_alert"] for row in selected),
            "feature_content_alerts": sum(row["baseline_feature_alert"] for row in selected),
            "histogram_or_mean_alerts": sum(row["oracle_histogram_alert"] is True for row in selected),
            "privileged_direct_comparator_alerts": sum(row["oracle_direct_label_alert"] is True for row in selected),
        }

    return {
        "phase": protocol["protocol_phase"],
        "protocol_id": protocol["id"],
        "protocol_sha256": hashlib.sha256(raw).hexdigest(),
        "environment": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "scikit_learn": sklearn.__version__,
        },
        "n_datasets": len(protocol["datasets"]),
        "n_cases": len(rows),
        "by_case": by_case,
        "records": rows,
        "limits": protocol["analysis"],
    }


if __name__ == "__main__":
    print(json.dumps(run_benchmark(), ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False))
