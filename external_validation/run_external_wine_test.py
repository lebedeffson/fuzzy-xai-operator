#!/usr/bin/env python3
from __future__ import annotations

import json
import zipfile
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.datasets import load_wine
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from fuzzyxai import build_proof_trace, build_route, render_dashboard, verify_proof_trace
from fuzzyxai.adapters.tabular_classification import TabularClassificationAdapter
from fuzzyxai.viz import save_proof_trace_json, save_route_json


ROOT = Path(__file__).resolve().parent
OUT = ROOT / "outputs"
TARGET_PROBABILITY = 0.68
ZIP_NAME = "external_wine_blackbox_validation.zip"


def _split_wine() -> tuple[Any, Any, Any, Any, Any]:
    data = load_wine()
    x_train, x_test, y_train, y_test = train_test_split(
        data.data,
        data.target,
        test_size=0.35,
        random_state=7,
        stratify=data.target,
    )
    return data, x_train, x_test, y_train, y_test


def _pick_moderate_object(probabilities: np.ndarray) -> int:
    confidence = probabilities.max(axis=1)
    candidates = [
        (abs(float(value) - TARGET_PROBABILITY), idx)
        for idx, value in enumerate(confidence)
        if 0.40 <= float(value) <= 0.90
    ]
    if candidates:
        return min(candidates)[1]
    return int(np.argmin(np.abs(confidence - TARGET_PROBABILITY)))


def _top_k_importance(names: list[str], importances: np.ndarray, top_k: int) -> dict[str, float]:
    total = float(np.sum(importances)) or 1.0
    normalized = importances / total
    order = np.argsort(normalized)[::-1][:top_k]
    return {names[index]: round(float(normalized[index]), 6) for index in order}


def _payload_for_model(model_key: str) -> dict[str, Any]:
    data, x_train, x_test, y_train, _ = _split_wine()
    names = list(data.feature_names)
    if model_key == "logistic_regression":
        model = make_pipeline(
            StandardScaler(),
            LogisticRegression(max_iter=5000, C=0.35, random_state=7),
        )
        top_k = 5
    elif model_key == "gradient_boosting":
        model = GradientBoostingClassifier(
            n_estimators=18,
            learning_rate=0.08,
            max_depth=1,
            random_state=7,
        )
        top_k = 2
    else:
        raise ValueError(f"unknown external wine model: {model_key}")

    model.fit(x_train, y_train)
    probabilities = model.predict_proba(x_test)
    sample_index = _pick_moderate_object(probabilities)
    selected_proba = probabilities[sample_index]
    predicted_class = int(selected_proba.argmax())
    class_probability = float(selected_proba[predicted_class])

    if model_key == "logistic_regression":
        coefficients = np.abs(model.named_steps["logisticregression"].coef_[predicted_class])
        raw_importance = coefficients
        model_name = "LogisticRegression"
    else:
        raw_importance = model.feature_importances_
        model_name = "GradientBoostingClassifier"

    return {
        "scenario_id": "external_wine_classification",
        "source_type": "tabular",
        "model_name": model_name,
        "dataset_name": "sklearn_wine",
        "predicted_class": predicted_class,
        "class_probability": round(class_probability, 6),
        "feature_values": {
            name: float(value)
            for name, value in zip(names, x_test[sample_index], strict=True)
        },
        "feature_importance": _top_k_importance(names, raw_importance, top_k),
        "quality_metrics": {
            "missing_rate": 0.0,
            "feature_range_violation": 0.0,
        },
        "context_values": {
            "external_task": True,
            "model_key": model_key,
            "top_k_importance": top_k,
            "sample_index": int(sample_index),
            "train_size": int(len(x_train)),
            "test_size": int(len(x_test)),
        },
    }


def _run_one(model_key: str) -> dict[str, Any]:
    payload = _payload_for_model(model_key)
    adapter = TabularClassificationAdapter()
    adapted = adapter.to_adapted_input(payload)
    route = build_route(adapted)
    trace = build_proof_trace(route)
    verification = verify_proof_trace(trace)
    if not verification.valid:
        raise SystemExit(f"{model_key}: proof verification failed: {verification.errors}")

    prefix = f"external_wine_{model_key}"
    route_path = save_route_json(route, OUT / f"{prefix}_route.json")
    proof_path = save_proof_trace_json(trace, OUT / f"{prefix}_proof_trace.json")
    dashboard_path = render_dashboard(route, OUT / f"{prefix}_operator_dashboard.png")
    computed = route.computed_result
    if computed.get("gamma", 0.0) <= 0 or computed.get("delta", 0.0) <= 0 or computed.get("rho", 0.0) <= 0:
        raise SystemExit(f"{model_key}: expected non-zero gamma/delta/rho, got {computed}")
    if route.final_action != "lower_confidence":
        raise SystemExit(f"{model_key}: expected lower_confidence, got {route.final_action}")

    summary = {
        "task": "sklearn_wine_classification",
        "package": "fuzzyxai",
        "model_key": model_key,
        "model_name": payload["model_name"],
        "scenario_id": route.scenario_id,
        "action": route.final_action,
        "diagnostic": computed.get("diagnostic_id"),
        "computed_result": computed,
        "verifier": "passed" if verification.valid else "failed",
        "source_commit": route.source_commit,
        "route": route_path.as_posix(),
        "proof_trace": proof_path.as_posix(),
        "dashboard": dashboard_path.as_posix(),
    }
    summary_path = OUT / f"{prefix}_summary.json"
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return summary


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    for old_file in OUT.glob("external_wine_*"):
        old_file.unlink()
    validations = [_run_one("logistic_regression"), _run_one("gradient_boosting")]
    aggregate = {
        "task": "sklearn_wine_classification",
        "package": "fuzzyxai",
        "scenario_id": "external_wine_classification",
        "models": [item["model_name"] for item in validations],
        "validations": validations,
        "verifier": "passed" if all(item["verifier"] == "passed" for item in validations) else "failed",
        "source_commit": validations[0]["source_commit"] if validations else "unknown",
    }
    summary_path = OUT / "external_wine_summary.json"
    summary_path.write_text(json.dumps(aggregate, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    zip_path = OUT / ZIP_NAME
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for path in sorted(OUT.glob("external_wine_*")):
            if path.name != ZIP_NAME and path.is_file():
                archive.write(path, arcname=path.name)
    print(json.dumps(aggregate, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
