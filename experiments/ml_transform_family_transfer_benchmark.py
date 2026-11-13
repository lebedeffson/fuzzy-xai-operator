"""Cross-family transfer of the *unchanged* single-transform FuzzyXAI observer.

Developer-constructed, explicitly instrumented sklearn pipelines.
The scorer observes supplied records only and never invokes model methods.
Not an independent, blind or natural-fault study.
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
from fuzzyxai.output_provenance import OutputRecorder
from fuzzyxai.transform_provenance import TransformRecorder
from sklearn.cluster import KMeans
from sklearn.datasets import (
    load_breast_cancer,
    load_diabetes,
    load_digits,
    load_iris,
    load_linnerud,
    load_wine,
)
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import (
    MaxAbsScaler,
    MinMaxScaler,
    PowerTransformer,
    QuantileTransformer,
    RobustScaler,
    SplineTransformer,
    StandardScaler,
)
from sklearn.svm import SVC

PROTOCOL = Path(__file__).with_name("ml_transform_family_transfer_protocol_v1.json")
TEST_KEY = b"transfer-experiment-only-not-a-deployment-key"

DATASETS = {
    "digits": load_digits, "wine": load_wine,
    "breast_cancer": load_breast_cancer, "diabetes": load_diabetes,
    "linnerud": load_linnerud, "iris": load_iris,
}
TRANSFORMERS = {
    "PCA": PCA, "RobustScaler": RobustScaler,
    "QuantileTransformer": QuantileTransformer,
    "SplineTransformer": SplineTransformer,
    "MinMaxScaler": MinMaxScaler, "PowerTransformer": PowerTransformer,
    "MaxAbsScaler": MaxAbsScaler, "StandardScaler": StandardScaler,
}
MODELS = {
    "KNeighborsClassifier": KNeighborsClassifier,
    "SVC": SVC, "LogisticRegression": LogisticRegression,
    "Ridge": Ridge, "RandomForestClassifier": RandomForestClassifier,
    "GaussianNB": GaussianNB, "KMeans": KMeans,
}

COUNTED_CODES = (
    "transform_train_input", "transform_evaluation_input",
    "transform_preprocessor_fit_link", "transform_model_fit_link",
    "transform_predict_link", "transform_predict_proba_link",
    "fit_trace_preprocessor_isolation",
)
CASE_TARGET_CODE = {
    "changed_model_fit_values": "transform_model_fit_link",
    "changed_inference_values": "transform_predict_link",
    "changed_reported_raw_evaluation": "transform_evaluation_input",
    "preprocessor_fit_on_evaluation": "transform_preprocessor_fit_link",
}


def _dataset(descriptor: dict[str, Any], seed: int, fraction: float):
    data = DATASETS[descriptor["dataset"]]()
    X = np.asarray(data.data, dtype=float)
    y = None if descriptor["task"] == "clustering" else np.asarray(data.target)
    # Current ML output metrics support one regression target; the target
    # HMAC witness separately supports multi-output fits.
    if descriptor["dataset"] == "linnerud":
        y = np.asarray(y, dtype=float)[:, 0]
    indices = np.arange(len(X))
    train, test = train_test_split(
        indices, test_size=fraction, random_state=seed,
        stratify=y if descriptor["task"] == "classification" else None,
    )
    label = descriptor["name"]
    return (
        X[train].copy(), X[test].copy(),
        y[train].copy() if y is not None else None,
        y[test].copy() if y is not None else None,
        tuple(f"{label}:{int(v)}" for v in train),
        tuple(f"{label}:{int(v)}" for v in test),
    )


def _construct(typ: str, args: dict[str, Any], catalog: dict[str, Any]):
    if typ not in catalog:
        raise ValueError(f"Type {typ} not fixed in transfer protocol")
    return catalog[typ](**args)


def _one(
    descriptor: dict[str, Any],
    scenario: str,
    seed: int,
    alternate_seed: int,
    fraction: float,
) -> dict[str, Any]:
    task = descriptor["task"]
    source_seed = alternate_seed if scenario == "valid_retrained" else seed
    train_X, eval_X, train_y, eval_y, train_ids, eval_ids = _dataset(
        descriptor, source_seed, fraction,
    )
    if scenario == "valid_shifted_evaluation":
        eval_X = eval_X.copy()
        eval_X[:, 0] += 0.0375

    fit = FitRecorder(target_hmac_key=TEST_KEY)
    fit_X = (
        np.concatenate((train_X, eval_X))
        if scenario == "preprocessor_fit_on_evaluation" else train_X
    )
    fit_ids = (
        train_ids + eval_ids
        if scenario == "preprocessor_fit_on_evaluation" else train_ids
    )
    preprocessor = fit.fit(
        _construct(descriptor["transformer"], descriptor["transformer_args"], TRANSFORMERS),
        fit_X, row_ids=fit_ids, role="preprocessor", feature_space="raw",
    )
    transform = TransformRecorder()
    transformed_train = transform.transform(
        preprocessor, train_X, row_ids=train_ids, split="train",
    )
    transformed_eval = transform.transform(
        preprocessor, eval_X, row_ids=eval_ids, split="evaluation",
    )
    actual_fit_input = (
        transformed_train + 0.12
        if scenario == "changed_model_fit_values" else transformed_train
    )
    model = _construct(descriptor["estimator"], descriptor["estimator_args"], MODELS)
    if task == "clustering":
        model = fit.fit(
            model, actual_fit_input, role="model", row_ids=train_ids,
            feature_space="transformed",
        )
    else:
        model = fit.fit(
            model, actual_fit_input, train_y, role="model",
            row_ids=train_ids, feature_space="transformed",
        )
    actual_predict_input = (
        transformed_eval + 0.125
        if scenario == "changed_inference_values" else transformed_eval
    )
    output = OutputRecorder()
    prediction = output.predict(
        model, actual_predict_input, row_ids=eval_ids, feature_space="transformed",
    )
    probability = (
        output.predict_proba(
            model, actual_predict_input,
            row_ids=eval_ids, feature_space="transformed",
        ) if task == "classification" and hasattr(model, "predict_proba") else None
    )
    reported_eval = eval_X.copy()
    if scenario == "changed_reported_raw_evaluation":
        reported_eval[0, 0] += 0.0625

    evidence = MLRunEvidence(
        task=task, model_family=type(model).__name__,
        train_features=train_X, evaluation_features=reported_eval,
        train_targets=train_y, evaluation_targets=eval_y,
        predictions=prediction, probabilities=probability,
        probability_classes=(
            tuple(str(value) for value in model.classes_)
            if probability is not None else None
        ),
        train_ids=train_ids, evaluation_ids=eval_ids,
        model_fit_ids=train_ids, preprocessor_fit_ids=fit_ids,
        preprocessor_applied=True,
        fit_records=[r.to_dict() for r in fit.records],
        output_records=[r.to_dict() for r in output.records],
        transform_records=(
            None if scenario == "missing_transform_witness"
            else [r.to_dict() for r in transform.records]
        ),
        model_version=f"{descriptor['name']}-seed{source_seed}",
        random_seed=source_seed,
    )

    observed = observe_ml_run(evidence, target_hmac_key=TEST_KEY)
    check = {entry.code: entry.status for entry in observed.checks}
    counted = {name: check.get(name) for name in COUNTED_CODES}
    bad = [code for code, status in counted.items() if status == "problem"]
    missing = scenario == "missing_transform_witness"

    # Metadata/scalar checks available before the value-level observer.
    baseline = bool(set(fit_ids) & set(eval_ids))
    # Direct comparator has EQUAL access to the same actual arrays used
    # in fit, transform and predict, not secretly privileged extra labels.
    oracle = (
        not np.array_equal(fit_X, train_X)
        or not np.array_equal(actual_fit_input, transformed_train)
        or not np.array_equal(actual_predict_input, transformed_eval)
        or not np.array_equal(reported_eval, eval_X)
    )
    expected = {
        "clean": "valid", "valid_shifted_evaluation": "valid",
        "valid_retrained": "valid",
        "changed_model_fit_values": "invalid",
        "changed_inference_values": "invalid",
        "changed_reported_raw_evaluation": "invalid",
        "preprocessor_fit_on_evaluation": "invalid",
        "missing_transform_witness": "unverifiable",
    }[scenario]
    return {
        "pipeline": descriptor["name"],
        "dataset": descriptor["dataset"],
        "task": task,
        "estimator": type(model).__name__,
        "transformer": type(preprocessor).__name__,
        "raw_feature_count": int(train_X.shape[1]),
        "transformed_feature_count": int(transformed_train.shape[1]),
        "scenario": scenario,
        "truth": expected,
        "checks": counted,
        "transform_trace": check.get("transform_trace"),
        "metadata_alert": baseline,
        "equal_evidence_direct_alert": oracle,
        "observer_alert": bool(bad),
        "observer_fault_codes": bad,
        "observer_missing": missing and check.get("transform_trace") == "unknown",
        "designated_check": CASE_TARGET_CODE.get(scenario),
        "designated_status": check.get(CASE_TARGET_CODE[scenario]) if scenario in CASE_TARGET_CODE else None,
    }


def run_benchmark() -> dict[str, Any]:
    raw = PROTOCOL.read_bytes()
    protocol = json.loads(raw)
    if protocol["schema_version"] != "ml-transform-family-transfer-development-v1":
        raise ValueError("Unrecognized transfer protocol")
    if len({row["name"] for row in protocol["pipelines"]}) != len(protocol["pipelines"]):
        raise ValueError("Duplicate pipeline name")
    seed, other = int(protocol["seed"]), int(protocol["alternative_seed"])
    records: list[dict[str, Any]] = []
    for descriptor in protocol["pipelines"]:
        for scenario in protocol["scenarios"]:
            records.append(
                _one(descriptor, scenario["name"], seed, other, float(protocol["evaluation_fraction"]))
            )
    by_scenario = {}
    for scenario in protocol["scenarios"]:
        name = scenario["name"]
        group = [row for row in records if row["scenario"] == name]
        by_scenario[name] = {
            "n": len(group),
            "metadata_alerts": sum(r["metadata_alert"] for r in group),
            "direct_alerts": sum(r["equal_evidence_direct_alert"] for r in group),
            "observer_alerts": sum(r["observer_alert"] for r in group),
            "observer_unknowns": sum(r["observer_missing"] for r in group),
        }
    return {
        "phase": protocol["phase"],
        "protocol_sha256": hashlib.sha256(raw).hexdigest(),
        "environment": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "scikit_learn": sklearn.__version__,
        },
        "n_pipelines": len(protocol["pipelines"]),
        "n_cases": len(records),
        "n_unique_source_datasets": len({d["dataset"] for d in protocol["pipelines"]}),
        "by_scenario": by_scenario,
        "records": records,
        "limitations": protocol["invariants"],
    }


if __name__ == "__main__":
    print(json.dumps(run_benchmark(), sort_keys=True, indent=2, allow_nan=False))
