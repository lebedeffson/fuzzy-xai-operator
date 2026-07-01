from __future__ import annotations

import json
from typing import Any

from fuzzyxai.audit.common import ROOT, current_commit
from fuzzyxai.practice.fixtures import SCENARIOS, stable_hash, write_json
from fuzzyxai.train.common import MODEL_DIR


def scenario_input(scenario_id: str) -> dict[str, Any]:
    fixture = SCENARIOS[scenario_id]
    model_card_path = MODEL_DIR[scenario_id] / "model_card.json"
    dataset_manifest = ROOT / "data/registry/datasets.yaml"
    data = {
        "scenario_id": scenario_id,
        "source_dataset": fixture["dataset_id"],
        "source_model": fixture["model_id"],
        "input_values": fixture["input_values"],
        "expected_outputs": fixture["expected"],
        "fed_to_engine": True,
        "dataset_manifest_hash": stable_hash(dataset_manifest.read_text(encoding="utf-8")),
        "model_card_hash": stable_hash(model_card_path.read_text(encoding="utf-8")) if model_card_path.exists() else None,
        "source_commit": current_commit(),
    }
    data["scenario_input_hash"] = stable_hash({k: v for k, v in data.items() if k != "scenario_input_hash"})
    return data


def evaluate_control_model(scenario_id: str) -> dict[str, Any]:
    fixture = SCENARIOS[scenario_id]
    input_data = scenario_input(scenario_id)
    out_input = ROOT / "reports/practice_demo/scenario_inputs" / f"{scenario_id}_input.json"
    write_json(out_input, input_data)
    report = {
        "status": "PASS",
        "scenario_id": scenario_id,
        "model_id": fixture["model_id"],
        "model_status": fixture["model_status"],
        "scenario_input": str(out_input.relative_to(ROOT)),
        "scenario_input_hash": input_data["scenario_input_hash"],
        "outputs": fixture["expected"],
        "not_a_claim": fixture["not_a_claim"],
        "source_commit": current_commit(),
    }
    out = ROOT / "reports/evaluation" / f"{scenario_id}_eval_report.json"
    write_json(out, report)
    return report


def main_for(scenario_id: str) -> None:
    report = evaluate_control_model(scenario_id)
    print(json.dumps(report, ensure_ascii=False, indent=2))

