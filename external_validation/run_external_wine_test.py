#!/usr/bin/env python3
from __future__ import annotations

import json
from pathlib import Path

from sklearn.datasets import load_wine
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

from fuzzyxai import build_proof_trace, build_route, render_dashboard, verify_proof_trace
from fuzzyxai.adapters.tabular_classification import TabularClassificationAdapter
from fuzzyxai.viz import save_proof_trace_json, save_route_json


ROOT = Path(__file__).resolve().parent
OUT = ROOT / "outputs"


def build_payload() -> dict:
    data = load_wine()
    x_train, x_test, y_train, _ = train_test_split(
        data.data,
        data.target,
        test_size=0.25,
        random_state=42,
        stratify=data.target,
    )
    model = RandomForestClassifier(n_estimators=120, random_state=42)
    model.fit(x_train, y_train)
    probabilities = model.predict_proba(x_test)
    best_index = max(range(len(probabilities)), key=lambda idx: max(probabilities[idx]))
    best_proba = probabilities[best_index]
    predicted_class = int(best_proba.argmax())
    class_probability = float(best_proba[predicted_class])
    feature_values = {
        name: float(value)
        for name, value in zip(data.feature_names, x_test[best_index], strict=True)
    }
    feature_importance = {
        name: float(value)
        for name, value in zip(data.feature_names, model.feature_importances_, strict=True)
    }
    return {
        "scenario_id": "external_wine_classification",
        "source_type": "tabular",
        "model_name": "RandomForestClassifier",
        "dataset_name": "sklearn_wine",
        "predicted_class": predicted_class,
        "class_probability": round(class_probability, 6),
        "feature_values": feature_values,
        "feature_importance": feature_importance,
        "quality_metrics": {
            "missing_rate": 0.0,
            "feature_range_violation": 0.0,
        },
        "context_values": {
            "external_task": True,
            "train_size": int(len(x_train)),
            "test_size": int(len(x_test)),
        },
    }


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    payload = build_payload()
    adapter = TabularClassificationAdapter()
    adapted = adapter.to_adapted_input(payload)
    route = build_route(adapted)
    trace = build_proof_trace(route)
    verification = verify_proof_trace(trace)
    if not verification.valid:
        raise SystemExit(f"proof verification failed: {verification.errors}")

    route_path = save_route_json(route, OUT / "external_wine_route.json")
    proof_path = save_proof_trace_json(trace, OUT / "external_wine_proof_trace.json")
    dashboard_path = render_dashboard(route, OUT / "external_wine_operator_dashboard.png")
    summary = {
        "task": "sklearn_wine_classification",
        "package": "fuzzyxai",
        "scenario_id": route.scenario_id,
        "action": route.final_action,
        "diagnostic": route.computed_result.get("diagnostic_id"),
        "computed_result": route.computed_result,
        "verifier": "passed" if verification.valid else "failed",
        "source_commit": route.source_commit,
        "route": route_path.as_posix(),
        "proof_trace": proof_path.as_posix(),
        "dashboard": dashboard_path.as_posix(),
    }
    summary_path = OUT / "external_wine_summary.json"
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
