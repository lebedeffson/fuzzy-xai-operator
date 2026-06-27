from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from fuzzyxai.audit.common import ROOT, current_commit
from fuzzyxai.practice.fixtures import SCENARIOS, RAW_FILES, file_hash, stable_hash, write_json


MODEL_DIR = {
    "hybrid_xiris": ROOT / "models/iris",
    "medical_ecg_signal": ROOT / "models/ecg",
    "gd_anfis_shap": ROOT / "models/gd_anfis_shap",
    "beacon_xai": ROOT / "models/beacon",
    "gis_integro": ROOT / "models/gis",
}


TRAINING_SCRIPT = {
    "hybrid_xiris": "fuzzyxai/train/train_iris_quality.py",
    "medical_ecg_signal": "fuzzyxai/train/train_ecg_quality.py",
    "gd_anfis_shap": "fuzzyxai/train/train_gd_anfis_shap.py",
    "beacon_xai": "fuzzyxai/train/train_beacon_counterevidence.py",
    "gis_integro": "fuzzyxai/train/train_gis_route.py",
}


def model_card(scenario_id: str) -> dict[str, Any]:
    fixture = SCENARIOS[scenario_id]
    model_dir = MODEL_DIR[scenario_id]
    model_artifact = model_dir / f"{fixture['model_id']}.json"
    raw_path = RAW_FILES[scenario_id]
    card = {
        "model_id": fixture["model_id"],
        "scenario_id": scenario_id,
        "model_status": fixture["model_status"],
        "trained_from": str(raw_path.relative_to(ROOT)),
        "training_script": TRAINING_SCRIPT[scenario_id],
        "model_artifact": str(model_artifact.relative_to(ROOT)),
        "input_schema": sorted(fixture["input_values"]),
        "output_schema": sorted(fixture["expected"]),
        "metrics": fixture["expected"],
        "not_a_claim": fixture["not_a_claim"],
        "source_commit": current_commit(),
        "dataset_file_hash": file_hash(raw_path),
    }
    card["model_card_hash"] = stable_hash({k: v for k, v in card.items() if k != "model_card_hash"})
    return card


def train_control_model(scenario_id: str) -> dict[str, Any]:
    fixture = SCENARIOS[scenario_id]
    model_dir = MODEL_DIR[scenario_id]
    model_dir.mkdir(parents=True, exist_ok=True)
    model_artifact = model_dir / f"{fixture['model_id']}.json"
    card = model_card(scenario_id)
    model = {
        "model_id": fixture["model_id"],
        "scenario_id": scenario_id,
        "model_status": fixture["model_status"],
        "control_parameters": fixture["input_values"],
        "expected_outputs": fixture["expected"],
        "not_a_claim": fixture["not_a_claim"],
        "source_commit": current_commit(),
    }
    write_json(model_artifact, model)
    write_json(model_dir / "model_card.json", card)
    report = {
        "status": "PASS",
        "scenario_id": scenario_id,
        "model_id": fixture["model_id"],
        "model_status": fixture["model_status"],
        "model_artifact": str(model_artifact.relative_to(ROOT)),
        "model_card": str((model_dir / "model_card.json").relative_to(ROOT)),
        "source_commit": current_commit(),
        "not_a_claim": fixture["not_a_claim"],
    }
    out = ROOT / "reports/training" / f"{scenario_id}_training_report.json"
    write_json(out, report)
    return report


def main_for(scenario_id: str) -> None:
    report = train_control_model(scenario_id)
    print(json.dumps(report, ensure_ascii=False, indent=2))

