#!/usr/bin/env python3
from __future__ import annotations

import csv
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PACKAGE_DIR = ROOT / "external_validation" / "outputs" / "external_wine_blackbox_validation"
MODEL_KEYS = ("logistic_regression", "gradient_boosting")


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def fail(errors: list[str]) -> int:
    print("operator-traceability-check: FAIL", file=sys.stderr)
    for error in errors:
        print(f"  - {error}", file=sys.stderr)
    return 1


def main() -> int:
    errors: list[str] = []
    rows: list[tuple[str, int, int, str, str]] = []
    for model_key in MODEL_KEYS:
        base = PACKAGE_DIR / model_key
        required = [
            "route.json",
            "operator_trace.json",
            "operator_table.csv",
            "proof_trace.json",
            "verifier_report.json",
            "dashboard_data.json",
            "operator_dashboard.png",
            "operator_dashboard_v2.png",
            "operator_dashboard_v2.html",
            "summary.json",
        ]
        for name in required:
            path = base / name
            if not path.exists() or path.stat().st_size == 0:
                errors.append(f"{model_key}: missing {name}")
        cards_dir = base / "operator_cards"
        if not cards_dir.exists() or not any(cards_dir.glob("*.md")):
            errors.append(f"{model_key}: missing operator_cards")
        if errors:
            continue

        route = load_json(base / "route.json")
        trace = load_json(base / "operator_trace.json")
        proof = load_json(base / "proof_trace.json")
        verifier = load_json(base / "verifier_report.json")
        dashboard = load_json(base / "dashboard_data.json")
        nodes = trace.get("nodes", [])
        edges = trace.get("edges", [])
        if verifier.get("overall_status") != "passed":
            errors.append(f"{model_key}: verifier_report not passed")
        if route.get("computed_result") != dashboard.get("computed_result"):
            errors.append(f"{model_key}: dashboard_data computed_result mismatch")
        if proof.get("computed_result") != route.get("computed_result"):
            errors.append(f"{model_key}: proof_trace computed_result mismatch")
        if not nodes:
            errors.append(f"{model_key}: no traced nodes")
        if not edges:
            errors.append(f"{model_key}: no traced edges")
        for node in nodes:
            node_id = node.get("node_id", "<unknown>")
            for key in ("input_values", "output_values", "components"):
                if not node.get(key):
                    errors.append(f"{model_key}:{node_id}: empty {key}")
            if not node.get("formula_text"):
                errors.append(f"{model_key}:{node_id}: missing formula_text")
            if not node.get("interpretation_ru"):
                errors.append(f"{model_key}:{node_id}: missing interpretation_ru")
            if node_id != "proof" and not node.get("next_node_ids"):
                errors.append(f"{model_key}:{node_id}: missing next_node_ids")
        for edge in edges:
            if not edge.get("passed_values"):
                errors.append(f"{model_key}:{edge.get('edge_id')}: empty passed_values")
        with (base / "operator_table.csv").open(encoding="utf-8") as file:
            table_rows = list(csv.DictReader(file))
        if len(table_rows) != len(nodes):
            errors.append(f"{model_key}: operator_table row count mismatch")
        card_count = len(list(cards_dir.glob("*.md")))
        if card_count < len(nodes):
            errors.append(f"{model_key}: operator_cards count mismatch")
        rows.append((model_key, len(nodes), len(edges), verifier.get("overall_status", ""), "yes"))

    if errors:
        return fail(errors)
    print("operator-traceability-check: PASS")
    for model_key, node_count, edge_count, verifier_status, dashboard_v2 in rows:
        print(f"  {model_key}: nodes={node_count}; edges={edge_count}; verifier={verifier_status}; dashboard_v2={dashboard_v2}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
