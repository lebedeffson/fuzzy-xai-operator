from __future__ import annotations

import csv
import html
import json
from hashlib import sha256
from pathlib import Path
from typing import Any

from fuzzyxai.core.types import OperatorRoute, ProofTrace
from fuzzyxai.proof.verifier import VerificationResult


def _json(data: Any) -> str:
    return json.dumps(data, ensure_ascii=False, sort_keys=True, indent=2)


def _hash(data: Any) -> str:
    return sha256(_json(data).encode("utf-8")).hexdigest()


def build_verifier_report(route: OperatorRoute, trace: ProofTrace, verification: VerificationResult) -> dict[str, Any]:
    computed = route.computed_result
    checks = [
        {"id": "source_commit_present", "status": "passed" if bool(route.source_commit and trace.source_commit) else "failed"},
        {"id": "route_proof_values_match", "status": "passed" if route.computed_result == trace.computed_result else "failed"},
        {"id": "action_matches_route", "status": "passed" if route.final_action == trace.final_action else "failed"},
        {"id": "diagnostic_present", "status": "passed" if bool(computed.get("diagnostic_id")) else "failed"},
        {"id": "operator_nodes_have_io", "status": "passed" if all(n.input_values and n.output_values for n in route.nodes) else "failed"},
        {"id": "operator_edges_have_passed_values", "status": "passed" if all(e.passed_values for e in route.edges) else "failed"},
    ]
    if {"gamma", "delta", "rho"} <= set(computed):
        gamma_expected = round(
            max(
                float(computed.get("uncertainty", computed.get("uncertainty_component", 0.0))),
                float(computed.get("quality_penalty", computed.get("quality_component", 0.0))),
                float(computed.get("conflict_component", 0.0)),
                float(computed.get("interval_component", 0.0)),
            ),
            6,
        )
        rho_expected = round(
            max(
                float(computed["gamma"]),
                float(computed["delta"]),
                float(computed.get("quality_component", computed.get("quality_penalty", 0.0))),
                float(computed.get("conflict_component", 0.0)),
                float(computed.get("interval_component", 0.0)),
            ),
            6,
        )
        checks.extend(
            [
                {"id": "gamma_matches_formula", "status": "passed" if computed["gamma"] == gamma_expected else "failed"},
                {"id": "rho_matches_formula", "status": "passed" if computed["rho"] == rho_expected else "failed"},
                {"id": "action_matches_risk_zone", "status": "passed" if _action_ok(computed["rho"], route.final_action) else "failed"},
            ]
        )
    checks.extend({"id": f"proof_verifier:{error}", "status": "failed"} for error in verification.errors)
    failed = [item for item in checks if item["status"] != "passed"]
    return {
        "overall_status": "passed" if verification.valid and not failed else "failed",
        "checks": checks,
        "errors": verification.errors + [item["id"] for item in failed],
        "warnings": [],
    }


def _action_ok(rho: float, action: str) -> bool:
    if rho < 0.35:
        return action == "accept"
    if rho < 0.60:
        return action == "lower_confidence"
    return action in {"audit", "defer_to_human", "audit_report"}


def route_to_operator_trace(route: OperatorRoute, verifier_report: dict[str, Any]) -> dict[str, Any]:
    route_dict = route.to_dict()
    return {
        "route_id": route_dict.get("route_id"),
        "scenario_id": route.scenario_id,
        "scenario_title_ru": route_dict.get("scenario_title_ru"),
        "source_commit": route.source_commit,
        "created_at": route_dict.get("created_at"),
        "nodes": route_dict["nodes"],
        "edges": route_dict["edges"],
        "computed_result": route.computed_result,
        "final_diagnostic_id": route_dict.get("final_diagnostic_id"),
        "final_action_id": route_dict.get("final_action_id"),
        "verification_summary": verifier_report,
    }


def build_dashboard_data(route: OperatorRoute, trace: ProofTrace, verifier_report: dict[str, Any]) -> dict[str, Any]:
    route_dict = route.to_dict()
    return {
        "route_id": route_dict.get("route_id"),
        "scenario_id": route.scenario_id,
        "title": route.title,
        "scenario_title_ru": route_dict.get("scenario_title_ru"),
        "source_commit": route.source_commit,
        "computed_result": route.computed_result,
        "nodes": route_dict["nodes"],
        "edges": route_dict["edges"],
        "verifier_report": verifier_report,
        "proof_hash": _hash(trace.to_dict()),
        "route_hash": _hash(route_dict),
    }


def write_json(path: Path, data: Any) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(_json(data) + "\n", encoding="utf-8")
    return path


def write_operator_table(route: OperatorRoute, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "order",
        "node_id",
        "title_ru",
        "operator_type",
        "input_summary",
        "formula",
        "components",
        "output",
        "thresholds",
        "status",
        "interpretation_ru",
        "next_node_ids",
    ]
    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        for index, node in enumerate(route.nodes, start=1):
            writer.writerow(
                {
                    "order": index,
                    "node_id": node.node_id,
                    "title_ru": node.title_ru or node.title,
                    "operator_type": node.operator_type or node.node_id,
                    "input_summary": _compact(node.input_values) or node.input_summary,
                    "formula": node.formula_text or node.formula_ref,
                    "components": _compact(node.components),
                    "output": _compact(node.output_values) or node.value,
                    "thresholds": _compact(node.thresholds),
                    "status": node.status,
                    "interpretation_ru": node.interpretation_ru,
                    "next_node_ids": ",".join(node.next_node_ids),
                }
            )
    return path


def _compact(value: Any) -> str:
    if value in (None, {}, []):
        return ""
    text = json.dumps(value, ensure_ascii=False, sort_keys=True)
    return text if len(text) <= 500 else text[:497] + "..."


def write_operator_cards(route: OperatorRoute, directory: Path) -> list[Path]:
    directory.mkdir(parents=True, exist_ok=True)
    paths: list[Path] = []
    for node in route.nodes:
        path = directory / f"{node.node_id}.md"
        body = "\n".join(
            [
                f"# {node.title_ru or node.title}",
                "",
                "## Input",
                _markdown_dict(node.input_values),
                "",
                "## Formula",
                node.formula_text or node.formula_ref or "Нормализация внешнего входа",
                "",
                "## Components",
                _markdown_dict(node.components),
                "",
                "## Output",
                _markdown_dict(node.output_values),
                "",
                "## Thresholds",
                _markdown_dict(node.thresholds),
                "",
                "## Status",
                f"{node.status}: {node.status_reason_ru}",
                "",
                "## Interpretation",
                node.interpretation_ru,
                "",
                "## Next",
                ", ".join(node.next_node_ids) or "final",
                "",
            ]
        )
        path.write_text(body, encoding="utf-8")
        paths.append(path)
    return paths


def _markdown_dict(value: dict[str, Any]) -> str:
    if not value:
        return "n/a"
    return "\n".join(f"- `{key}` = `{json.dumps(val, ensure_ascii=False)}`" for key, val in value.items())


def render_dashboard_v2(dashboard_data: dict[str, Any], output_path: Path) -> Path:
    try:
        import matplotlib.pyplot as plt
    except Exception as exc:  # pragma: no cover
        raise RuntimeError("matplotlib is required for dashboard v2 rendering") from exc

    output_path.parent.mkdir(parents=True, exist_ok=True)
    nodes = dashboard_data["nodes"]
    computed = dashboard_data["computed_result"]
    checks = dashboard_data["verifier_report"]["checks"]
    fig, axes = plt.subplots(2, 2, figsize=(18, 12))
    fig.patch.set_facecolor("#f7f7f4")
    fig.suptitle(
        f"{dashboard_data['scenario_title_ru']} | source_commit: {dashboard_data['source_commit'][:12]}",
        fontsize=16,
        fontweight="bold",
    )
    ax_route, ax_values, ax_formulas, ax_verify = axes.flatten()
    for ax in axes.flatten():
        ax.axis("off")

    ax_route.set_title("Operator Route")
    y = 0.95
    for index, node in enumerate(nodes, start=1):
        ax_route.text(0.02, y, f"{index}. {node['node_id']} -> {node.get('next_node_ids') or ['final']}", fontsize=10, va="top")
        ax_route.text(0.06, y - 0.035, f"{node.get('title_ru') or node.get('title')}: {node.get('value')}", fontsize=9, va="top")
        y -= 0.10

    ax_values.set_title("Key Values")
    rows = [
        ("p", computed.get("class_probability")),
        ("gamma", computed.get("gamma")),
        ("delta", computed.get("delta")),
        ("rho", computed.get("rho")),
        ("diagnostic", computed.get("diagnostic_id")),
        ("action", computed.get("action_id") or computed.get("action")),
    ]
    ax_values.table(cellText=[[k, v] for k, v in rows], colLabels=["key", "value"], loc="center", cellLoc="left")

    ax_formulas.set_title("Formulas")
    formula_lines = [f"{node['node_id']}: {node.get('formula_text')}" for node in nodes if node.get("formula_text")]
    ax_formulas.text(0.02, 0.95, "\n".join(formula_lines[:12]), fontsize=9, va="top")

    ax_verify.set_title("Proof / Verifier")
    verify_lines = [
        f"route_id: {dashboard_data['route_id']}",
        f"route_hash: {dashboard_data['route_hash'][:16]}",
        f"proof_hash: {dashboard_data['proof_hash'][:16]}",
        "",
        *[f"{item['id']}: {item['status']}" for item in checks[:10]],
    ]
    ax_verify.text(0.02, 0.95, "\n".join(verify_lines), fontsize=9, va="top")
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return output_path


def write_dashboard_html(dashboard_data: dict[str, Any], output_path: Path) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    data = html.escape(_json(dashboard_data))
    rows = "\n".join(
        f"<tr><td>{html.escape(node['node_id'])}</td><td>{html.escape(node.get('formula_text') or '')}</td><td><pre>{html.escape(_json(node.get('output_values', {})))}</pre></td><td>{html.escape(node.get('interpretation_ru') or '')}</td></tr>"
        for node in dashboard_data["nodes"]
    )
    doc = f"""<!doctype html>
<html lang="ru"><meta charset="utf-8"><title>FuzzyXAI Operator Dashboard v2</title>
<style>body{{font-family:Arial,sans-serif;margin:24px;color:#17202a}}table{{border-collapse:collapse;width:100%}}td,th{{border:1px solid #ccc;padding:8px;vertical-align:top}}pre{{white-space:pre-wrap;margin:0}}.meta{{background:#f3f5f7;padding:12px}}</style>
<h1>{html.escape(dashboard_data['scenario_title_ru'])}</h1>
<div class="meta">route_id: {html.escape(str(dashboard_data['route_id']))}<br>source_commit: {html.escape(str(dashboard_data['source_commit']))}<br>verifier: {html.escape(str(dashboard_data['verifier_report']['overall_status']))}</div>
<h2>Key Values</h2><pre>{html.escape(_json(dashboard_data['computed_result']))}</pre>
<h2>Operator Cards</h2><table><tr><th>node</th><th>formula</th><th>output</th><th>interpretation</th></tr>{rows}</table>
<h2>Embedded Data</h2><script type="application/json" id="dashboard-data">{data}</script>
</html>
"""
    output_path.write_text(doc, encoding="utf-8")
    return output_path


def write_traceability_artifacts(route: OperatorRoute, trace: ProofTrace, verification: VerificationResult, directory: Path) -> dict[str, Path]:
    directory.mkdir(parents=True, exist_ok=True)
    verifier_report = build_verifier_report(route, trace, verification)
    operator_trace = route_to_operator_trace(route, verifier_report)
    dashboard_data = build_dashboard_data(route, trace, verifier_report)
    paths = {
        "operator_trace": write_json(directory / "operator_trace.json", operator_trace),
        "operator_table": write_operator_table(route, directory / "operator_table.csv"),
        "verifier_report": write_json(directory / "verifier_report.json", verifier_report),
        "dashboard_data": write_json(directory / "dashboard_data.json", dashboard_data),
        "operator_dashboard_v2": render_dashboard_v2(dashboard_data, directory / "operator_dashboard_v2.png"),
        "operator_dashboard_v2_html": write_dashboard_html(dashboard_data, directory / "operator_dashboard_v2.html"),
    }
    write_operator_cards(route, directory / "operator_cards")
    return paths
