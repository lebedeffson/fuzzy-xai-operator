#!/usr/bin/env python3
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SCENARIOS = {
    "hybrid_xiris": ("block", "D_quality_source_conflict"),
    "medical_ecg_signal": ("defer_to_human", "D_signal_quality"),
    "gd_anfis_shap": ("audit", "D_rule_attribution_conflict"),
    "beacon_xai": ("audit", "D_counterevidence_conflict"),
    "gis_integro": ("audit_report", "D_route_context_limit"),
}
SITE_FORBIDDEN = [
    re.compile(r"from\s+fuzzyxai\b"),
    re.compile(r"import\s+fuzzyxai\b"),
    re.compile(r"gamma\s*="),
    re.compile(r"delta\s*="),
    re.compile(r"rho\s*="),
]


def load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def fail(errors: list[str], message: str) -> None:
    errors.append(message)


def diagnostic_ids(proof: dict) -> set[str]:
    ids = set()
    for item in proof.get("diagnostics", []) or []:
        if isinstance(item, dict) and item.get("diagnostic_id"):
            ids.add(str(item["diagnostic_id"]))
    computed = proof.get("computed_result", {}) or {}
    if computed.get("diagnostic_id"):
        ids.add(str(computed["diagnostic_id"]))
    return ids


def check_scenario(scenario_id: str, expected_action: str, expected_diagnostic: str, errors: list[str]) -> None:
    base = ROOT / "applications" / "scenarios" / scenario_id
    paths = {
        "route": base / "route" / "route.json",
        "proof": base / "proof" / "proof_trace.json",
        "source_proof": base / "proof" / f"{scenario_id}_proof_package.json",
        "figure": base / "figures" / "operator_dashboard.png",
        "payload": base / "site_payload" / "scenario.json",
        "site_route": ROOT / "site" / "dubnaxai" / "public" / "routes" / f"{scenario_id}_route.json",
        "site_figure": ROOT / "site" / "dubnaxai" / "public" / "figures" / f"{scenario_id}_operator_dashboard.png",
    }
    for label, path in paths.items():
        if not path.exists():
            fail(errors, f"{scenario_id}: missing {label}: {path}")
        elif path.is_file() and path.stat().st_size == 0:
            fail(errors, f"{scenario_id}: empty {label}: {path}")
    if any(not path.exists() for path in paths.values()):
        return

    route = load(paths["route"])
    proof = load(paths["proof"])
    source_proof = load(paths["source_proof"])
    payload = load(paths["payload"])
    site_route = load(paths["site_route"])

    if route.get("scenario_id") != scenario_id:
        fail(errors, f"{scenario_id}: route scenario_id mismatch")
    if route.get("final_action") != expected_action:
        fail(errors, f"{scenario_id}: route action {route.get('final_action')} != {expected_action}")
    if proof.get("final_action") != expected_action:
        fail(errors, f"{scenario_id}: proof action {proof.get('final_action')} != {expected_action}")
    if source_proof != proof:
        fail(errors, f"{scenario_id}: proof_trace.json differs from source proof package")
    if proof.get("verifier_status") != "PASS" or route.get("verifier_status") != "PASS":
        fail(errors, f"{scenario_id}: verifier status is not PASS")
    if expected_diagnostic not in diagnostic_ids(proof):
        fail(errors, f"{scenario_id}: diagnostic {expected_diagnostic} missing")
    if len(route.get("nodes", []) or []) < 5:
        fail(errors, f"{scenario_id}: route has too few nodes")
    route_node_ids = {node.get("node_id") for node in route.get("nodes", [])}
    for required in {"input_artifact", "explanation_object", "diagnostics", "action", "proof"}:
        if required not in route_node_ids:
            fail(errors, f"{scenario_id}: route missing node {required}")
    if payload.get("route") != "route/route.json":
        fail(errors, f"{scenario_id}: payload route link mismatch")
    if payload.get("proof_trace") != "proof/proof_trace.json":
        fail(errors, f"{scenario_id}: payload proof link mismatch")
    if payload.get("figure") != "figures/operator_dashboard.png":
        fail(errors, f"{scenario_id}: payload figure link mismatch")
    if payload.get("final_action") != expected_action:
        fail(errors, f"{scenario_id}: payload action mismatch")
    if site_route != route:
        fail(errors, f"{scenario_id}: site route differs from application route")


def check_site_is_display_only(errors: list[str]) -> None:
    for path in (ROOT / "site" / "dubnaxai").rglob("*"):
        if not path.is_file() or path.suffix not in {".py", ".js", ".jsx", ".ts", ".tsx"}:
            continue
        text = path.read_text(encoding="utf-8")
        for pattern in SITE_FORBIDDEN:
            if pattern.search(text):
                fail(errors, f"site computes/imports forbidden FuzzyXAI token in {path}: {pattern.pattern}")


def main() -> None:
    errors: list[str] = []
    for scenario_id, (action, diagnostic) in SCENARIOS.items():
        check_scenario(scenario_id, action, diagnostic, errors)
    check_site_is_display_only(errors)
    if errors:
        print("operator-route-check: FAIL", file=sys.stderr)
        for error in errors:
            print(f"- {error}", file=sys.stderr)
        raise SystemExit(1)
    print("operator-route-check: PASS")


if __name__ == "__main__":
    main()
