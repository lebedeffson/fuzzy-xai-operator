#!/usr/bin/env python3
from __future__ import annotations

import json
import shutil
import zipfile
from datetime import UTC, datetime
from hashlib import sha256
from pathlib import Path
from typing import Any

import fuzzyxai
import numpy as np
from sklearn.datasets import load_wine
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from fuzzyxai import build_proof_trace, build_route, render_dashboard, verify_proof_trace
from fuzzyxai.adapters.tabular_classification import TabularClassificationAdapter
from fuzzyxai.viz import save_proof_trace_json, save_route_json, write_traceability_artifacts


ROOT = Path(__file__).resolve().parent
OUT = ROOT / "outputs"
TARGET_PROBABILITY = 0.68
ZIP_NAME = "external_wine_blackbox_validation.zip"
PACKAGE_NAME = "external_wine_blackbox_validation"
PACKAGE_DIR = OUT / PACKAGE_NAME


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


def _sha256(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _model_dir_name(model_key: str) -> str:
    return model_key


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
    model_dir = PACKAGE_DIR / _model_dir_name(model_key)
    model_dir.mkdir(parents=True, exist_ok=True)
    route_path = save_route_json(route, OUT / f"{prefix}_route.json")
    proof_path = save_proof_trace_json(trace, OUT / f"{prefix}_proof_trace.json")
    dashboard_path = render_dashboard(route, OUT / f"{prefix}_operator_dashboard.png")
    package_route = model_dir / "route.json"
    package_proof = model_dir / "proof_trace.json"
    package_dashboard = model_dir / "operator_dashboard.png"
    shutil.copy2(route_path, package_route)
    shutil.copy2(proof_path, package_proof)
    shutil.copy2(dashboard_path, package_dashboard)
    write_traceability_artifacts(route, trace, verification, model_dir)
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
        "route": f"{_model_dir_name(model_key)}/route.json",
        "proof_trace": f"{_model_dir_name(model_key)}/proof_trace.json",
        "dashboard": f"{_model_dir_name(model_key)}/operator_dashboard.png",
        "operator_trace": f"{_model_dir_name(model_key)}/operator_trace.json",
        "operator_table": f"{_model_dir_name(model_key)}/operator_table.csv",
        "verifier_report": f"{_model_dir_name(model_key)}/verifier_report.json",
        "dashboard_data": f"{_model_dir_name(model_key)}/dashboard_data.json",
        "dashboard_v2": f"{_model_dir_name(model_key)}/operator_dashboard_v2.png",
        "dashboard_v2_html": f"{_model_dir_name(model_key)}/operator_dashboard_v2.html",
        "operator_cards": f"{_model_dir_name(model_key)}/operator_cards/",
    }
    package_summary = model_dir / "summary.json"
    package_summary.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    shutil.copy2(package_summary, OUT / f"{prefix}_summary.json")
    return summary


def _write_report(aggregate: dict[str, Any]) -> Path:
    rows = []
    for item in aggregate["validations"]:
        computed = item["computed_result"]
        rows.append(
            "| {model} | {p:.6f} | {gamma:.6f} | {delta:.6f} | {rho:.6f} | {action} | {diagnostic} |".format(
                model=item["model_name"],
                p=float(computed["class_probability"]),
                gamma=float(computed["gamma"]),
                delta=float(computed["delta"]),
                rho=float(computed["rho"]),
                action=item["action"],
                diagnostic=item["diagnostic"],
            )
        )
    report = "\n".join(
        [
            "# External FuzzyXAI Black-Box Validation",
            "",
            f"- task: `{aggregate['task']}`",
            f"- scenario_id: `{aggregate['scenario_id']}`",
            f"- source_commit: `{aggregate['source_commit']}`",
            f"- verifier: `{aggregate['verifier']}`",
            f"- package: `{ZIP_NAME}`",
            "",
            "The package was generated from an installed `fuzzyxai` framework import and does not use `applications/scenarios`.",
            "Both checks use moderate-confidence wine-classification objects and top-k feature importances, so operator values are non-zero.",
            "",
            "| Model | p | gamma | delta | rho | action | diagnostic |",
            "|---|---:|---:|---:|---:|---|---|",
            *rows,
            "",
            "Formulas:",
            "",
            "- `gamma = max(1 - class_probability, quality_penalty)`",
            "- `delta = 1 - sum(top_k_feature_importance)`",
            "- `rho = max(gamma, delta)`",
            "",
        ]
    )
    path = PACKAGE_DIR / "external_validation_report.md"
    path.write_text(report, encoding="utf-8")
    return path


def _write_provenance(aggregate: dict[str, Any]) -> Path:
    provenance = {
        "task": aggregate["task"],
        "scenario_id": aggregate["scenario_id"],
        "source_commit": aggregate["source_commit"],
        "generated_at_utc": datetime.now(UTC).isoformat(),
        "cwd": Path.cwd().as_posix(),
        "fuzzyxai_file": Path(fuzzyxai.__file__).resolve().as_posix(),
        "package_boundary_ok": "framework/fuzzyxai/fuzzyxai" in Path(fuzzyxai.__file__).resolve().as_posix(),
        "applications_used": False,
        "imports": [
            "fuzzyxai",
            "fuzzyxai.adapters.tabular_classification.TabularClassificationAdapter",
            "sklearn.datasets.load_wine",
            "sklearn.linear_model.LogisticRegression",
            "sklearn.ensemble.GradientBoostingClassifier",
        ],
    }
    path = PACKAGE_DIR / "import_provenance.json"
    path.write_text(json.dumps(provenance, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return path


def _write_manifest(aggregate: dict[str, Any]) -> Path:
    files = []
    for path in sorted(PACKAGE_DIR.rglob("*")):
        if path.is_file() and path.name != "manifest.json":
            files.append(
                {
                    "path": path.relative_to(PACKAGE_DIR).as_posix(),
                    "size_bytes": path.stat().st_size,
                    "sha256": _sha256(path),
                }
            )
    manifest = {
        "package_name": PACKAGE_NAME,
        "task": aggregate["task"],
        "scenario_id": aggregate["scenario_id"],
        "source_commit": aggregate["source_commit"],
        "verifier": aggregate["verifier"],
        "models": aggregate["models"],
        "files": files,
    }
    path = PACKAGE_DIR / "manifest.json"
    path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return path


def _build_zip() -> Path:
    zip_path = OUT / ZIP_NAME
    if zip_path.exists():
        zip_path.unlink()
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for path in sorted(PACKAGE_DIR.rglob("*")):
            if path.is_file():
                archive.write(path, arcname=(Path(PACKAGE_NAME) / path.relative_to(PACKAGE_DIR)).as_posix())
    return zip_path


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    for old_file in OUT.glob("external_wine_*"):
        if old_file.is_file():
            old_file.unlink()
        elif old_file.is_dir():
            shutil.rmtree(old_file)
    PACKAGE_DIR.mkdir(parents=True, exist_ok=True)
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
    shutil.copy2(summary_path, PACKAGE_DIR / "external_wine_summary.json")
    _write_report(aggregate)
    _write_provenance(aggregate)
    _write_manifest(aggregate)
    _build_zip()
    print(json.dumps(aggregate, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
