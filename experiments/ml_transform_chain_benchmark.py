"""Developmental, explicitly instrumented StandardScaler chain comparison."""

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
from fuzzyxai.output_provenance import OutputRecorder
from fuzzyxai.transform_provenance import TransformRecorder
from sklearn.cluster import KMeans
from sklearn.datasets import load_breast_cancer, load_diabetes, load_iris, load_wine
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

PROTOCOL = Path(__file__).with_name("ml_transform_chain_protocol_v1.json")
KEY = b"benchmark-only-target-hmac-key-not-for-deployment"


def _model(name: str, seed: int) -> Any:
    if name == "LogisticRegression":
        return LogisticRegression(solver="newton-cholesky", max_iter=800, random_state=seed)
    if name == "Ridge":
        return Ridge()
    if name == "KMeans":
        return KMeans(n_clusters=3, random_state=seed, n_init=10)
    raise ValueError("Unexpected estimator not in frozen protocol")


def _source(name: str, task: str, seed: int):
    dataset = {
        "breast_cancer": load_breast_cancer,
        "wine": load_wine,
        "diabetes": load_diabetes,
        "iris_clustering": load_iris,
    }[name]()
    features = np.asarray(dataset.data, dtype=float)
    target = None if task == "clustering" else np.asarray(dataset.target)
    training, evaluation = train_test_split(
        np.arange(len(features)), random_state=seed, test_size=0.25,
        stratify=target if task == "classification" else None,
    )
    return (
        features[training].copy(), features[evaluation].copy(),
        target[training] if target is not None else None,
        target[evaluation] if target is not None else None,
        tuple(f"{name}:{int(v)}" for v in training),
        tuple(f"{name}:{int(v)}" for v in evaluation),
    )


def _checks(report: Any) -> dict[str, str]:
    return {item.code: item.status for item in report.checks}


def run_benchmark() -> dict[str, Any]:
    original_bytes = PROTOCOL.read_bytes()
    protocol = json.loads(original_bytes)
    rows: list[dict[str, Any]] = []
    seed = int(protocol["seed"])

    for descriptor in protocol["pipelines"]:
        name, task, estimator_name = descriptor["name"], descriptor["task"], descriptor["estimator"]
        raw_train, raw_eval, y_train, y_eval, train_ids, eval_ids = _source(name, task, seed)
        for desc in protocol["scenarios"]:
            case = desc["name"]
            fit = FitRecorder(target_hmac_key=KEY)
            fit_input_preproc = (
                np.concatenate([raw_train, raw_eval])
                if case == "preprocessor_fit_on_evaluation" else raw_train
            )
            preproc_fit_ids = (
                train_ids + eval_ids
                if case == "preprocessor_fit_on_evaluation" else train_ids
            )
            scaler = fit.fit(
                StandardScaler(), fit_input_preproc,
                row_ids=preproc_fit_ids, role="preprocessor", feature_space="raw",
            )
            transform = TransformRecorder()
            x_train = transform.transform(scaler, raw_train, split="train", row_ids=train_ids)
            x_eval = transform.transform(scaler, raw_eval, split="evaluation", row_ids=eval_ids)
            actual_model_input = x_train + 0.2 if case == "changed_transformed_model_fit" else x_train
            model = _model(estimator_name, seed)
            model = fit.fit(
                model, actual_model_input, y_train, role="model",
                row_ids=train_ids, feature_space="transformed",
            )
            actual_predict_input = x_eval + 0.15 if case == "changed_transformed_inference" else x_eval
            outputs = OutputRecorder()
            predictions = outputs.predict(model, actual_predict_input, row_ids=eval_ids, feature_space="transformed")
            probabilities = (
                outputs.predict_proba(model, actual_predict_input, row_ids=eval_ids, feature_space="transformed")
                if task == "classification" else None
            )
            source_eval_report = raw_eval.copy()
            if case == "changed_declared_raw_evaluation":
                source_eval_report[0, 0] += 0.15
            elif case not in {
                "clean", "changed_transformed_model_fit", "changed_transformed_inference",
                "missing_transform_witness", "preprocessor_fit_on_evaluation",
            }:
                raise ValueError(f"Unrecognized frozen scenario {case}")

            evidence = MLRunEvidence(
                task=task, model_family=type(model).__name__,
                train_features=raw_train, evaluation_features=source_eval_report,
                train_targets=y_train, evaluation_targets=y_eval,
                predictions=predictions, probabilities=probabilities,
                probability_classes=tuple(str(x) for x in model.classes_) if task == "classification" else None,
                train_ids=train_ids, evaluation_ids=eval_ids,
                model_fit_ids=train_ids, preprocessor_fit_ids=preproc_fit_ids,
                preprocessor_applied=True,
                fit_records=[record.to_dict() for record in fit.records],
                output_records=[record.to_dict() for record in outputs.records],
                transform_records=(
                    None if case == "missing_transform_witness"
                    else [record.to_dict() for record in transform.records]
                ),
                model_version=f"{name}-seed{seed}", random_seed=seed,
            )
            report = observe_ml_run(evidence, target_hmac_key=KEY)
            status = _checks(report)
            expected_codes = (
                "transform_trace", "transform_train_input", "transform_evaluation_input",
                "transform_preprocessor_fit_link", "transform_model_fit_link",
                "transform_predict_link", "transform_predict_proba_link",
                "fit_trace_preprocessor_isolation", "fit_trace_model_targets",
            )
            # Numeric/schema/ID baseline has no link to the actual transform
            # contents. Honest preprocessor-fit IDs remain detectable.
            baseline = bool(set(preproc_fit_ids) & set(eval_ids))
            oracle = (
                not np.array_equal(fit_input_preproc, raw_train)
                or not np.array_equal(actual_model_input, x_train)
                or not np.array_equal(actual_predict_input, x_eval)
                or not np.array_equal(source_eval_report, raw_eval)
            )
            rows.append({
                "dataset": name, "task": task, "estimator": type(model).__name__,
                "declared_estimator": estimator_name, "case": case,
                "metadata_baseline_alert": baseline,
                "equal_evidence_direct_comparison_alert": oracle,
                "checks": {code: status.get(code) for code in expected_codes},
            })

    aggregate: dict[str, dict[str, Any]] = {}
    for desc in protocol["scenarios"]:
        case = desc["name"]
        subset = [row for row in rows if row["case"] == case]
        aggregate[case] = {
            "n": len(subset),
            "metadata_baseline_alerts": sum(row["metadata_baseline_alert"] for row in subset),
            "direct_comparator_alerts": sum(row["equal_evidence_direct_comparison_alert"] for row in subset),
            "transform_problem": sum(
                "problem" in [status for code, status in row["checks"].items() if code.startswith("transform_")]
                for row in subset
            ),
            "transform_missing_unknown": sum(row["checks"]["transform_trace"] == "unknown" for row in subset),
        }

    return {
        "protocol": protocol["id"],
        "phase": protocol["phase"],
        "protocol_sha256": hashlib.sha256(original_bytes).hexdigest(),
        "environment": {
            "python": platform.python_version(),
            "numpy": np.__version__, "scikit_learn": sklearn.__version__,
        },
        "n_pipelines": len(protocol["pipelines"]),
        "n_cases": len(rows),
        "by_scenario": aggregate,
        "records": rows,
        "limits": protocol["guardrails"],
    }


if __name__ == "__main__":
    print(json.dumps(run_benchmark(), indent=2, ensure_ascii=False, sort_keys=True, allow_nan=False))
