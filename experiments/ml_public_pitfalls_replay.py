"""Replay independently documented sklearn pitfalls with the unchanged observer.

Not a blinded external incident benchmark. sklearn's Common Pitfalls guide
authored the correct/incorrect patterns; this repo implements and instruments
each pattern. Labels are developer-known. Same-access array comparison is the
fair detector baseline; abstention on missing witness is required.
"""

from __future__ import annotations

import hashlib
import json
import platform
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
import sklearn
from sklearn.datasets import load_diabetes, make_regression
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.feature_selection import SelectKBest
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import accuracy_score, mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from fuzzyxai.fit_provenance import FitRecorder
from fuzzyxai.ml_observer import MLRunEvidence, observe_ml_run
from fuzzyxai.output_provenance import OutputRecorder
from fuzzyxai.transform_provenance import TransformRecorder

PROTOCOL = Path(__file__).with_name("ml_public_pitfalls_protocol_v1.json")
KEY = b"external-pattern-pilot-key-not-for-deployment"
COUNTED = (
    "split_isolation", "preprocessor_fit_isolation",
    "fit_trace_preprocessor_isolation", "transform_train_input",
    "transform_evaluation_input", "transform_preprocessor_fit_link",
    "transform_model_fit_link", "transform_predict_link",
    "transform_predict_proba_link",
)
SCENARIOS = ("correct", "documented_error", "missing_transform_record")


def _source_case(name: str, *, seed: int, fraction: float):
    if name == "inconsistent_scaler":
        x, y = make_regression(
            n_samples=100, n_features=1, noise=1, random_state=seed,
        )
        preprocessor, estimator, task = StandardScaler(), LinearRegression(), "regression"
    elif name == "feature_selection_leak":
        rng = np.random.RandomState(seed)
        x = rng.standard_normal((200, 10000))
        y = rng.choice(2, 200)
        preprocessor = SelectKBest(k=25)
        estimator = HistGradientBoostingClassifier(random_state=1)
        task = "classification"
    elif name == "pre_split_scaler":
        x, y = load_diabetes(return_X_y=True)
        preprocessor, estimator, task = StandardScaler(), Ridge(alpha=1.0), "regression"
    else:
        raise ValueError(f"Unknown pre-registered example {name!r}")
    x = np.asarray(x, dtype=float)
    y = np.asarray(y)
    all_idx = np.arange(len(x))
    idx_train, idx_test = train_test_split(
        all_idx, random_state=seed, test_size=fraction,
    )
    train_x, eval_x = x[idx_train].copy(), x[idx_test].copy()
    train_y, eval_y = y[idx_train].copy(), y[idx_test].copy()
    train_ids = tuple(f"{name}:row:{int(i)}" for i in idx_train)
    eval_ids = tuple(f"{name}:row:{int(i)}" for i in idx_test)
    all_ids = tuple(f"{name}:row:{int(i)}" for i in all_idx)
    return {
        "train_x": train_x, "eval_x": eval_x,
        "train_y": train_y, "eval_y": eval_y,
        "all_x": x, "all_y": y, "all_ids": all_ids,
        "train_ids": train_ids, "eval_ids": eval_ids,
        "preprocessor": preprocessor, "estimator": estimator,
        "task": task,
    }


def _run_one(example: dict[str, Any], scenario: str) -> dict[str, Any]:
    name = example["id"]
    invalid = scenario == "documented_error"
    missing = scenario == "missing_transform_record"
    context = _source_case(
        name, seed=int(example["seed"]), fraction=float(example["test_fraction"]),
    )
    train_x, eval_x = context["train_x"], context["eval_x"]
    train_y, eval_y = context["train_y"], context["eval_y"]
    train_ids, eval_ids = context["train_ids"], context["eval_ids"]
    is_leaky_fit = invalid and name in ("feature_selection_leak", "pre_split_scaler")
    fit_x = context["all_x"] if is_leaky_fit else train_x
    fit_ids = context["all_ids"] if is_leaky_fit else train_ids
    is_selector = name == "feature_selection_leak"
    fit_y = (
        context["all_y"] if is_leaky_fit else train_y
    ) if is_selector else None
    fit = FitRecorder(target_hmac_key=KEY)
    preprocessor = fit.fit(
        context["preprocessor"], fit_x, fit_y, role="preprocessor",
        row_ids=fit_ids, feature_space="raw",
    )
    transform = TransformRecorder()
    transformed_train = transform.transform(
        preprocessor, train_x, split="train", row_ids=train_ids,
    )
    transformed_eval = transform.transform(
        preprocessor, eval_x, split="evaluation", row_ids=eval_ids,
    )
    model = fit.fit(
        context["estimator"], transformed_train, train_y,
        role="model", row_ids=train_ids, feature_space="transformed",
    )
    # The externally documented 12.1 error is executed as-is: model fitted on
    # scaled training values, but receives the raw evaluation feature vector.
    actual_input = (
        eval_x if invalid and name == "inconsistent_scaler"
        else transformed_eval
    )
    output = OutputRecorder()
    preds = output.predict(
        model, actual_input, row_ids=eval_ids, feature_space="transformed",
    )
    proba = (
        output.predict_proba(
            model, actual_input, row_ids=eval_ids, feature_space="transformed",
        ) if context["task"] == "classification" else None
    )
    evidence = MLRunEvidence(
        task=context["task"],
        model_family=type(model).__name__,
        train_features=train_x,
        evaluation_features=eval_x,
        train_targets=train_y,
        evaluation_targets=eval_y,
        predictions=preds,
        probabilities=proba,
        probability_classes=tuple(str(i) for i in model.classes_) if proba is not None else None,
        train_ids=train_ids, evaluation_ids=eval_ids,
        model_fit_ids=train_ids,
        preprocessor_fit_ids=fit_ids,
        preprocessor_applied=True,
        fit_records=[rec.to_dict() for rec in fit.records],
        transform_records=None if missing else [rec.to_dict() for rec in transform.records],
        output_records=[rec.to_dict() for rec in output.records],
        model_version=f"{name}:source-replay-v1",
        random_seed=int(example["seed"]),
    )
    obs = observe_ml_run(evidence, target_hmac_key=KEY)
    checks = {item.code: item.status for item in obs.checks}
    faults = sorted(k for k in COUNTED if checks.get(k) == "problem")
    trace_missing = checks.get("transform_trace") == "unknown"
    if missing and trace_missing:
        observer = "abstain"
    else:
        observer = "alert" if faults else "no_alert"

    # Compare the very same data/scope witnesses.  The "missing" scenario
    # WITHHOLDS transform evidence from both sides: neither may claim clean.
    if missing:
        direct = "abstain"
    else:
        direct_alert = (
            bool(set(fit_ids) & set(eval_ids))
            or not np.array_equal(fit_x, train_x)
            or not np.array_equal(transformed_train, transformed_train)
            or not np.array_equal(actual_input, transformed_eval)
        )
        direct = "alert" if direct_alert else "no_alert"
    # IDs only (no output/transform content access).
    metadata = "alert" if bool(set(fit_ids) & set(eval_ids)) else "no_alert"
    truth = (
        "unverifiable" if missing else
        "invalid" if invalid else "valid"
    )
    quality_name = "accuracy" if context["task"] == "classification" else "mse"
    quality = (
        float(accuracy_score(eval_y, preds)) if quality_name == "accuracy"
        else float(mean_squared_error(eval_y, preds))
    )
    return {
        "example": name, "source_section": example["source_section"],
        "scenario": scenario, "truth": truth, "rows_train": len(train_x),
        "rows_evaluation": len(eval_x), "model_type": type(model).__name__,
        "preprocessor_type": type(preprocessor).__name__,
        "preprocessor_fit_rows": len(fit_ids),
        "observer": observer, "metadata_only": metadata,
        "equal_evidence_direct": direct,
        "designated_check": checks.get(PROTOCOL_CHECKS[name]),
        "transform_trace": checks.get("transform_trace"),
        "counted_faults": faults,
        "model_metric": quality_name, "model_metric_value": quality,
        "limitation": "no provenance without witnessed transform calls" if missing else None,
    }


PROTOCOL_CHECKS = {
    "inconsistent_scaler": "transform_predict_link",
    "feature_selection_leak": "transform_preprocessor_fit_link",
    "pre_split_scaler": "transform_preprocessor_fit_link",
}


def _metrics(records: list[dict[str, Any]], key: str) -> dict[str, int]:
    valid = [r for r in records if r["truth"] == "valid"]
    invalid = [r for r in records if r["truth"] == "invalid"]
    unknown = [r for r in records if r["truth"] == "unverifiable"]
    return {
        "true_positive": sum(r[key] == "alert" for r in invalid),
        "false_negative": sum(r[key] != "alert" for r in invalid),
        "false_positive": sum(r[key] == "alert" for r in valid),
        "true_negative": sum(r[key] != "alert" for r in valid),
        "abstentions_on_missing": sum(r[key] == "abstain" for r in unknown),
        "unwarranted_alerts_on_missing": sum(r[key] == "alert" for r in unknown),
        "unwarranted_no_alert_on_missing": sum(r[key] == "no_alert" for r in unknown),
    }


def run_benchmark() -> dict[str, Any]:
    raw = PROTOCOL.read_bytes()
    protocol = json.loads(raw)
    if protocol.get("schema_version") != "fuzzyxai-public-sklearn-pitfalls-pilot-v1":
        raise ValueError("Protocol/schema version mismatch")
    examples = protocol["reproducers"]
    scenarios = tuple(x["id"] for x in protocol["scenarios"])
    if scenarios != SCENARIOS or {x["id"] for x in examples} != set(PROTOCOL_CHECKS):
        raise ValueError("Refusing a mutated pre-registered case catalog")
    records = [_run_one(e, scenario) for e in examples for scenario in scenarios]
    assert len(records) == protocol["anticipated_counts"]["executions"]
    assert Counter(r["truth"] for r in records) == {
        "valid": 3, "invalid": 3, "unverifiable": 3,
    }
    for row in records:
        if row["truth"] == "invalid":
            row["hit_designated_rule"] = row["designated_check"] == "problem"
        else:
            row["hit_designated_rule"] = None
    return {
        "phase": protocol["phase"],
        "protocol_sha256": hashlib.sha256(raw).hexdigest(),
        "source_url": protocol["source"]["url"],
        "source_kind": protocol["source"]["source_type"],
        "environment": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "scikit_learn": sklearn.__version__,
        },
        "n_external_examples": len(examples),
        "n_cases": len(records),
        "truth_counts": dict(Counter(r["truth"] for r in records)),
        "by_method": {
            "observer": _metrics(records, "observer"),
            "equal_evidence_direct": _metrics(records, "equal_evidence_direct"),
            "metadata_only": _metrics(records, "metadata_only"),
        },
        "records": records,
        "limits": protocol["known_limits"],
        "registered_rules": protocol["decision_rules"],
    }


if __name__ == "__main__":
    print(json.dumps(run_benchmark(), sort_keys=True, indent=2, allow_nan=False))
