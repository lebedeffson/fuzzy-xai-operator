"""Developmental benchmark: read-only ML lifecycle observer vs ordinary checks.

The experiment *trains* a few sklearn models only to generate independent
observations and deliberate broken pipelines. The observer itself NEVER fits
or predicts. See ml_lifecycle_development_protocol_v1.json for fixed cases.
Not a blinded evaluation or a claim of unknown-error detection.
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
from fuzzyxai.ml_observer import MLObservation, MLRunEvidence, observe_ml_run
from sklearn.cluster import KMeans
from sklearn.datasets import load_breast_cancer, load_diabetes, load_iris, load_wine
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.utils.validation import check_array

PROTOCOL_FILE = Path(__file__).with_name("ml_lifecycle_development_protocol_v1.json")


def _baseline_numeric(evidence: MLRunEvidence) -> set[str]:
    """Ordinary sklearn array and output validation, without provenance."""

    problems: set[str] = set()
    matrices: dict[str, np.ndarray] = {}
    for name, value in (
        ("train_features", evidence.train_features),
        ("evaluation_features", evidence.evaluation_features),
    ):
        if value is None:
            continue
        try:
            matrices[name] = check_array(value, dtype=float)
        except (TypeError, ValueError):
            problems.add(name)
    train = matrices.get("train_features")
    eval_data = matrices.get("evaluation_features")
    if train is not None and eval_data is not None and train.shape[1] != eval_data.shape[1]:
        problems.add("feature_width")
    if evidence.predictions is not None and eval_data is not None and len(evidence.predictions) != len(eval_data):
        problems.add("predictions")
    if evidence.task == "classification" and evidence.probabilities is not None:
        try:
            probabilities = np.asarray(evidence.probabilities, dtype=float)
            if (
                probabilities.ndim != 2
                or probabilities.shape[1] < 2
                or (eval_data is not None and len(probabilities) != len(eval_data))
                or not np.isfinite(probabilities).all()
                or not ((probabilities >= 0.0) & (probabilities <= 1.0)).all()
                or not np.allclose(probabilities.sum(axis=1), 1.0, atol=1e-5, rtol=0)
            ):
                problems.add("probabilities")
        except (TypeError, ValueError):
            problems.add("probabilities")
    return problems




def _baseline_label_metrics(evidence: MLRunEvidence) -> dict[str, float]:
    """Independently recompute ordinary sklearn evaluation metrics."""

    if evidence.task == "clustering" or evidence.evaluation_targets is None or evidence.predictions is None:
        return {}
    truth, predicted = np.asarray(evidence.evaluation_targets), np.asarray(evidence.predictions)
    if truth.ndim != 1 or predicted.ndim != 1 or len(truth) != len(predicted) or not len(truth):
        return {}
    if evidence.task == "classification":
        return {
            "accuracy": float(accuracy_score(truth, predicted)),
            "macro_f1": float(f1_score(truth, predicted, average="macro", zero_division=0)),
        }
    observed = np.asarray(truth, dtype=float)
    values = np.asarray(predicted, dtype=float)
    output = {
        "mae": float(mean_absolute_error(observed, values)),
        "rmse": float(np.sqrt(np.mean((observed - values) ** 2))),
    }
    if len(observed) >= 2 and np.var(observed) > 0.0:
        output["r2"] = float(r2_score(observed, values))
    return output


def _baseline_provenance(evidence: MLRunEvidence) -> set[str]:
    """Competent contract baseline: numeric checks plus all declared IDs/schema."""

    problems = _baseline_numeric(evidence)
    if evidence.train_ids is not None and evidence.evaluation_ids is not None and set(evidence.train_ids) & set(evidence.evaluation_ids):
        problems.add("split_isolation")
    if evidence.evaluation_ids is not None:
        for code, fit_ids in (
            ("model_fit_isolation", evidence.model_fit_ids),
            ("preprocessor_fit_isolation", evidence.preprocessor_fit_ids),
        ):
            if fit_ids is not None and set(evidence.evaluation_ids) & set(fit_ids):
                problems.add(code)
    if (
        evidence.train_feature_names is not None
        and evidence.evaluation_feature_names is not None
        and evidence.train_feature_names != evidence.evaluation_feature_names
    ):
        problems.add("ordered_feature_schema")
    return problems


def _fit_with_recorded_ids(estimator: Any, X: np.ndarray, row_ids: tuple[str, ...], y: np.ndarray | None) -> tuple[Any, tuple[str, ...]]:
    """Record the IDs at the actual fit invocation rather than infer them."""

    if len(row_ids) != len(X):
        raise ValueError("actual fit source IDs must agree with source rows")
    if y is None:
        estimator.fit(X)
    else:
        estimator.fit(X, y)
    return estimator, tuple(row_ids)


def _build_evidence(
    name: str,
    task: str,
    model_family: str,
    x_train: np.ndarray,
    x_test: np.ndarray,
    y_train: np.ndarray | None,
    y_test: np.ndarray | None,
    train_ids: tuple[str, ...],
    test_ids: tuple[str, ...],
    names: tuple[str, ...],
    model: Any,
    model_fit_ids: tuple[str, ...],
    scaler: StandardScaler | None,
    preprocessor_fit_ids: tuple[str, ...] | None,
    seed: int,
) -> MLRunEvidence:
    transformed = scaler.transform(x_test) if scaler is not None else x_test
    predictions = model.predict(transformed)
    probabilities = model.predict_proba(transformed) if task == "classification" else None
    classes = tuple(str(x) for x in model.classes_) if task == "classification" else None
    return MLRunEvidence(
        task=task,
        model_family=model_family,
        train_features=x_train,
        evaluation_features=x_test,
        train_targets=y_train,
        evaluation_targets=y_test,
        predictions=predictions,
        probabilities=probabilities,
        probability_classes=classes,
        train_feature_names=names,
        evaluation_feature_names=names,
        train_ids=train_ids,
        evaluation_ids=test_ids,
        model_fit_ids=model_fit_ids,
        preprocessor_fit_ids=preprocessor_fit_ids,
        model_version=f"{name}-seed{seed}",
        random_seed=seed,
    )


def _runs(name: str, task: str, seed: int) -> tuple[MLRunEvidence, dict[str, MLRunEvidence]]:
    ds = {
        "breast_cancer": load_breast_cancer,
        "wine": load_wine,
        "diabetes": load_diabetes,
        "iris": load_iris,
    }[name]()
    X = np.asarray(ds.data, dtype=float)
    y = None if task == "clustering" else np.asarray(ds.target)
    train_idx, test_idx = train_test_split(
        np.arange(len(X)),
        test_size=0.25,
        random_state=seed,
        stratify=y if task == "classification" else None,
    )
    train, test = X[train_idx].copy(), X[test_idx].copy()
    train_y = y[train_idx].copy() if y is not None else None
    test_y = y[test_idx].copy() if y is not None else None
    train_ids = tuple(f"{name}-{i}" for i in train_idx)
    test_ids = tuple(f"{name}-{i}" for i in test_idx)
    names = tuple(str(a) for a in getattr(ds, "feature_names", [f"feature_{k}" for k in range(X.shape[1])]))
    classifier = task == "classification"
    has_scaler = name == "breast_cancer"

    def make_estimator() -> Any:
        if name == "breast_cancer":
            return LogisticRegression(max_iter=1500, random_state=seed)
        if name == "wine":
            return RandomForestClassifier(n_estimators=40, random_state=seed)
        if name == "diabetes":
            return RandomForestRegressor(n_estimators=40, random_state=seed)
        return KMeans(n_clusters=3, random_state=seed, n_init=10)

    scaler: StandardScaler | None = None
    pre_fit: tuple[str, ...] | None = None
    if has_scaler:
        scaler, pre_fit = _fit_with_recorded_ids(StandardScaler(), train, train_ids, None)
    model_inputs = scaler.transform(train) if scaler is not None else train
    model, fit_ids = _fit_with_recorded_ids(make_estimator(), model_inputs, train_ids, train_y)
    model_family = type(model).__name__
    clean = _build_evidence(name, task, model_family, train, test, train_y, test_y, train_ids, test_ids, names, model, fit_ids, scaler, pre_fit, seed)

    scenarios: dict[str, MLRunEvidence] = {"clean": clean}

    # A metadata-level overlap, useful to test declared provenance.
    changed_ids = (train_ids[0],) + test_ids[1:]
    scenarios["split_row_id_overlap"] = replace(clean, evaluation_ids=changed_ids)

    # A REAL feature-axis mismatch (not just an edited field name).
    swapped = test.copy()
    swapped[:, [0, 1]] = swapped[:, [1, 0]]
    swapped_names = list(names)
    swapped_names[0], swapped_names[1] = swapped_names[1], swapped_names[0]
    swapped_ev = _build_evidence(
        name, task, model_family, train, swapped, train_y, test_y, train_ids,
        test_ids, names, model, fit_ids, scaler, pre_fit, seed,
    )
    scenarios["feature_columns_permuted"] = replace(swapped_ev, evaluation_feature_names=tuple(swapped_names))

    # Numeric feature content copied with a NEW identifier; only a warning.
    copied = test.copy()
    copied[0] = train[0]
    scenarios["copied_numeric_row_new_id"] = _build_evidence(
        name, task, model_family, train, copied, train_y, test_y, train_ids,
        test_ids, names, model, fit_ids, scaler, pre_fit, seed,
    )

    # Input shift followed by actual prediction from the original model.
    shifted = test.copy()
    train_scale = max(float(np.std(train[:, 0])), 0.01)
    shifted[:, 0] = shifted[:, 0] + 8.0 * train_scale
    scenarios["numeric_mean_shift"] = _build_evidence(
        name, task, model_family, train, shifted, train_y, test_y, train_ids,
        test_ids, names, model, fit_ids, scaler, pre_fit, seed,
    )

    scenarios["missing_source_identifiers"] = replace(
        clean, train_ids=None, evaluation_ids=None, model_fit_ids=None, preprocessor_fit_ids=None
    )

    if classifier:
        damaged = np.asarray(clean.probabilities, dtype=float).copy()
        damaged[0, 0] = damaged[0, 0] + 0.2
        scenarios["invalid_probability_rows"] = replace(clean, probabilities=damaged)

    if task != "clustering":
        assert test_y is not None and train_y is not None
        scenarios["evaluation_labels_permuted"] = replace(clean, evaluation_targets=np.roll(test_y, 1))
        # Intentional training contamination: actual .fit sees evaluation
        # labels. Its evaluation accuracy MUST NOT be used as generalization.
        full_x = np.vstack([train, test])
        full_y = np.concatenate([train_y, test_y])
        full_ids = train_ids + test_ids
        raw_fit = scaler.transform(full_x) if scaler is not None else full_x
        leaked_model, leaked_fit_ids = _fit_with_recorded_ids(make_estimator(), raw_fit, full_ids, full_y)
        scenarios["model_fit_includes_evaluation"] = _build_evidence(
            name, task, model_family, train, test, train_y, test_y, train_ids,
            test_ids, names, leaked_model, leaked_fit_ids, scaler, pre_fit, seed,
        )

    if has_scaler:
        # A real preprocessing leak: fitted on the union, then estimator
        # trained only on original training subset.
        full_ids = train_ids + test_ids
        leaked_scaler, leaked_pre_ids = _fit_with_recorded_ids(
            StandardScaler(), np.vstack([train, test]), full_ids, None
        )
        leaked_features = leaked_scaler.transform(train)
        leaked_estimator, model_ids = _fit_with_recorded_ids(
            make_estimator(), leaked_features, train_ids, train_y
        )
        scenarios["preprocessor_fit_includes_evaluation"] = _build_evidence(
            name, task, model_family, train, test, train_y, test_y, train_ids,
            test_ids, names, leaked_estimator, model_ids, leaked_scaler,
            leaked_pre_ids, seed,
        )

    return clean, scenarios


def _code_status(report: MLObservation, code: str) -> str | None:
    found = [check.status for check in report.checks if check.code == code]
    if len(found) > 1:
        raise ValueError(f"duplicate observer check: {code}")
    return found[0] if found else None


def run_benchmark() -> dict[str, Any]:
    """Execute the predeclared, developmental four-dataset evaluation."""

    raw = PROTOCOL_FILE.read_bytes()
    protocol = json.loads(raw)
    seed = int(protocol["seed"])
    rows: list[dict[str, Any]] = []
    for descriptor in protocol["datasets"]:
        name = descriptor["name"]
        task = descriptor["task"]
        _, scenarios = _runs(name, task, seed)
        for case_name, evidence in scenarios.items():
            report = observe_ml_run(evidence)
            target = next(case for case in protocol["scenarios"] if case["name"] == case_name)
            expected = target.get("expected_observer", "")
            intended_code = expected.split(":")[0] if ":" in expected else None
            rows.append(
                {
                    "dataset": name,
                    "task": task,
                    "case": case_name,
                    "declared_scenario_type": target["category"],
                    "target_check": intended_code,
                    "target_status": _code_status(report, intended_code) if intended_code is not None else None,
                    "plain_sklearn_flags": sorted(_baseline_numeric(evidence)),
                    "baseline_sklearn_metrics": _baseline_label_metrics(evidence),
                    "metadata_baseline_flags": sorted(_baseline_provenance(evidence)),
                    "observer_flags": [
                        {"code": check.code, "status": check.status}
                        for check in report.checks if check.status in ("problem", "warning")
                    ],
                    "observer_unknown": [
                        check.code for check in report.checks if check.status == "unknown"
                    ],
                    "observer_overall": report.overall_status,
                    "observer_metrics": report.metrics,
                }
            )

    by_case: dict[str, dict[str, Any]] = {}
    for descriptor in protocol["scenarios"]:
        case = descriptor["name"]
        sample = [item for item in rows if item["case"] == case]
        by_case[case] = {
            "n_pipelines_applicable": len(sample),
            "plain_baseline_alerts": sum(bool(item["plain_sklearn_flags"]) for item in sample),
            "metadata_baseline_alerts": sum(bool(item["metadata_baseline_flags"]) for item in sample),
            "observer_alerts": sum(bool(item["observer_flags"]) for item in sample),
            "target_check": descriptor.get("expected_observer"),
            "target_outcomes": [
                {"dataset": row["dataset"], "status": row["target_status"]}
                for row in sample
            ],
        }

    return {
        "protocol_id": protocol["id"],
        "protocol_sha256": hashlib.sha256(raw).hexdigest(),
        "phase": "developmental_known_faults_not_external_validation",
        "environment": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "scikit_learn": sklearn.__version__,
        },
        "n_pipelines": len(protocol["datasets"]),
        "n_cases": len(rows),
        "by_case": by_case,
        "cases": rows,
        "scientific_limits": protocol["analysis_constraints"],
    }


if __name__ == "__main__":
    print(json.dumps(run_benchmark(), ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False))
