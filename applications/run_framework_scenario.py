from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
FRAMEWORK = ROOT / "framework" / "fuzzyxai"
if str(FRAMEWORK) not in sys.path:
    sys.path.insert(0, str(FRAMEWORK))

from fuzzyxai import build_proof_trace, build_route, render_dashboard, verify_proof_trace
from fuzzyxai.examples import load_example
from fuzzyxai.viz import save_proof_trace_json, save_route_json


def run_framework_scenario(scenario_id: str, scenario_dir: Path) -> dict:
    adapted = load_example(scenario_id)
    route = build_route(adapted)
    trace = build_proof_trace(route)
    verification = verify_proof_trace(trace)
    if not verification.valid:
        raise SystemExit(f"{scenario_id}: proof trace failed: {verification.errors}")

    route_path = save_route_json(route, scenario_dir / "route" / "route.json")
    proof_path = save_proof_trace_json(trace, scenario_dir / "proof" / "proof_trace.json")
    figure_path = render_dashboard(route, scenario_dir / "figures" / "operator_dashboard.png")

    payload = {
        "scenario_id": scenario_id,
        "title_ru": route.title,
        "route": "route/route.json",
        "route_path": "route/route.json",
        "proof_trace": "proof/proof_trace.json",
        "proof_path": "proof/proof_trace.json",
        "figure": "figures/operator_dashboard.png",
        "dashboard_path": "figures/operator_dashboard.png",
        "action_id": route.final_action,
        "diagnostic_id": route.computed_result.get("diagnostic_id", ""),
        "final_action": route.final_action,
        "verifier_status": "PASS",
        "source": "fuzzyxai public API",
    }
    payload_path = scenario_dir / "site_payload" / "scenario.json"
    payload_path.parent.mkdir(parents=True, exist_ok=True)
    payload_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    result = {
        "scenario_id": scenario_id,
        "status": "PASS",
        "expected_action": route.final_action,
        "final_action": route.final_action,
        "computed_result": route.computed_result,
        "diagnostics": [item.get("diagnostic_id") for item in route.diagnostics],
        "verifier_status": "PASS",
        "route": route_path.as_posix(),
        "proof_trace": proof_path.as_posix(),
        "dashboard": figure_path.as_posix(),
    }
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return result
