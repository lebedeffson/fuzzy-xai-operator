#!/usr/bin/env python3
from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
FRAMEWORK = ROOT / "framework" / "fuzzyxai"
if str(FRAMEWORK) not in sys.path:
    sys.path.insert(0, str(FRAMEWORK))

from fuzzyxai.viz import load_route_from_proof, render_operator_dashboard


SCENARIOS = [
    "hybrid_xiris",
    "medical_ecg_signal",
    "gd_anfis_shap",
    "beacon_xai",
    "gis_integro",
]


def export_scenario(scenario_id: str) -> dict[str, str]:
    scenario_dir = ROOT / "applications" / "scenarios" / scenario_id
    proof_path = scenario_dir / "proof" / f"{scenario_id}_proof_package.json"
    route = load_route_from_proof(proof_path)

    route_path = scenario_dir / "route" / "route.json"
    proof_trace_path = scenario_dir / "proof" / "proof_trace.json"
    figure_path = scenario_dir / "figures" / "operator_dashboard.png"
    site_payload_path = scenario_dir / "site_payload" / "scenario.json"

    route.write_json(route_path)
    proof_trace_path.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(proof_path, proof_trace_path)
    render_operator_dashboard(route, figure_path)

    payload = {
        "scenario_id": scenario_id,
        "route": route_path.relative_to(scenario_dir).as_posix(),
        "proof_trace": proof_trace_path.relative_to(scenario_dir).as_posix(),
        "figure": figure_path.relative_to(scenario_dir).as_posix(),
        "final_action": route.final_action,
        "verifier_status": route.verifier_status,
        "source_commit": route.source_commit,
    }
    site_payload_path.parent.mkdir(parents=True, exist_ok=True)
    site_payload_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    report_route = ROOT / "reports" / "routes" / f"{scenario_id}_route.json"
    report_figure = ROOT / "reports" / "figures" / f"{scenario_id}_operator_dashboard.png"
    site_route = ROOT / "site" / "dubnaxai" / "public" / "routes" / f"{scenario_id}_route.json"
    site_figure = ROOT / "site" / "dubnaxai" / "public" / "figures" / f"{scenario_id}_operator_dashboard.png"
    for target, source in [
        (report_route, route_path),
        (report_figure, figure_path),
        (site_route, route_path),
        (site_figure, figure_path),
    ]:
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)

    return {
        "scenario_id": scenario_id,
        "route": route_path.as_posix(),
        "proof_trace": proof_trace_path.as_posix(),
        "figure": figure_path.as_posix(),
        "site_payload": site_payload_path.as_posix(),
    }


def main() -> None:
    outputs = [export_scenario(scenario_id) for scenario_id in SCENARIOS]
    summary = ROOT / "reports" / "routes" / "operator_routes_summary.json"
    summary.parent.mkdir(parents=True, exist_ok=True)
    summary.write_text(json.dumps(outputs, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(summary)


if __name__ == "__main__":
    main()
