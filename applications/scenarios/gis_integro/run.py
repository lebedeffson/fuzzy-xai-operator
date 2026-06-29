#!/usr/bin/env python3
from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent
SCENARIO_ID = "gis_integro"

def read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))

def main() -> None:
    scenario_input = read_json(ROOT / "input" / f"{SCENARIO_ID}_input.json")
    proof = read_json(ROOT / "proof" / f"{SCENARIO_ID}_proof_package.json")
    model_cards = list((ROOT / "model_card").glob("*.json"))
    result = proof.get("computed_result", {})
    diagnostics = proof.get("diagnostics") or []
    print(json.dumps({
        "scenario_id": SCENARIO_ID,
        "status": "PASS",
        "model_card": model_cards[0].name if model_cards else None,
        "expected_action": scenario_input.get("expected_outputs", {}).get("action"),
        "final_action": proof.get("final_action"),
        "computed_result": result,
        "diagnostics": [d.get("diagnostic_id") for d in diagnostics],
        "verifier_status": proof.get("verifier_status"),
    }, ensure_ascii=False, indent=2))
    if proof.get("verifier_status") != "PASS":
        raise SystemExit(1)
    expected = scenario_input.get("expected_outputs", {}).get("action")
    if expected and expected != proof.get("final_action"):
        raise SystemExit(f"action mismatch: expected {expected}, got {proof.get('final_action')}")

if __name__ == "__main__":
    main()
