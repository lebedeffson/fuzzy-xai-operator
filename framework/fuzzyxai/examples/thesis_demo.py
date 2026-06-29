from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any, Dict, List

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from fuzzyxai import ExplainPlan, Rule, SystemOperator, Trace, compose, interpretability_index, interpretability_loss
from fuzzyxai.core.trust_evaluator import entropy_from_memberships, semantic_disagreement
from fuzzyxai.hierarchy import HesitantFS, IntervalFS, MultiLevelFS, NeutrosophicFS
from fuzzyxai.selection.profile_builder import build_profile
from fuzzyxai.selection.pareto_selector import Candidate, select_minimal_sufficient
from fuzzyxai.selection.compatibility import synthesize_levels
from fuzzyxai.visual.interactive_graph import save_composition_html
from fuzzyxai.text.explanation_generator import generate_human_explanation

OUT = ROOT / "reports"
OUT.mkdir(exist_ok=True)


def candidates() -> List[Candidate]:
    return [
        Candidate("F0", {"u_num", "u_ling"}, 0.10, 0.03, 0.42),
        Candidate("FI", {"u_num", "u_ling", "u_int"}, 0.22, 0.05, 0.30),
        Candidate("FH", {"u_num", "u_ling", "u_exp"}, 0.30, 0.81, 0.25),
        Candidate("FNsrc", {"u_num", "u_ling", "u_if", "u_conf", "u_trace"}, 0.42, 0.09, 0.18),
        Candidate("FML-user", {"u_num", "u_ling", "u_int", "u_exp", "u_conf", "u_multi"}, 0.60, 0.99, 0.08),
        Candidate("FML-audit", {"u_num", "u_ling", "u_int", "u_exp", "u_conf", "u_trace", "u_multi"}, 0.68, 1.00, 0.04),
    ]


def run_demo() -> Dict[str, Any]:
    timeline: List[Dict[str, Any]] = []
    plan = ExplainPlan().with_reduction_weight(0.10)
    timeline.append({"step": 1, "name": "Initialize ExplainPlan", "payload": {"beta": plan.beta, "i_min": plan.i_min}})

    metadata = {
        "has_intervals": True,
        "num_experts": 2,
        "source_conflict": True,
        "requires_audit": True,
        "multi_level": True,
    }
    profile = build_profile(metadata)
    selected = select_minimal_sufficient(profile, candidates(), mode="audit")
    timeline.append({"step": 2, "name": "Build P_sit and select class", "payload": {"profile": sorted(profile), "selected": selected.name}})

    risk = 0.72
    interval = IntervalFS(lambda x: 0.68, lambda x: 0.76, policy="mid")
    hesitant = HesitantFS(lambda x: [0.61, 0.78])
    neutro = NeutrosophicFS(lambda x: 0.78, lambda x: 0.20, lambda x: 0.64, source_t="risk_model", source_f="expert_panel")
    multi = MultiLevelFS(
        [interval, hesitant, neutro],
        gamma={("interval", "hesitant", "same_case"), ("neutrosophic", "source", "conflict")},
        weights=[0.25, 0.25, 0.50],
    )
    _, delta = multi.reduce_to_f0()
    timeline.append({"step": 3, "name": "Construct A_k^F and reduce", "payload": {"representation": "FML-audit", "delta": round(delta, 6)}})

    op = SystemOperator(plan)
    rules_model = [
        Rule("r_high_check", {"risk": "high"}, "additional_check"),
        Rule("r_medium_watch", {"risk": "medium"}, "watch"),
    ]
    e_model = op.explain_scalar_risk(
        risk,
        rules_model,
        Trace("case-demo-model", "v1", "2026-05-22T10:00:00", source="thesis_demo", checksum="demo-model"),
        model_uncertainty=0.08,
        trace_uncertainty=0.02,
    )
    e_model.representation = multi
    e_model.reduction_loss = delta
    mu = e_model.metadata["memberships"]
    H = entropy_from_memberships([mu["low"], mu["medium"], mu["high"]], epsilon=plan.epsilon)
    timeline.append({"step": 4, "name": "Build E_k^ext", "payload": {"risk": risk, "memberships": mu, "H": round(H, 6), "uncertainty": round(e_model.uncertainty, 6)}})

    rules_decision = [Rule("r_decision_high", {"risk": "high"}, "send_to_check")]
    e_decision = op.explain_scalar_risk(
        0.74,
        rules_decision,
        Trace("case-demo-decision", "v1", "2026-05-22T10:00:01", source="thesis_demo", checksum="demo-decision"),
        model_uncertainty=0.10,
        trace_uncertainty=0.01,
    )
    gamma = semantic_disagreement(e_model, e_decision, plan.beta)
    composed = compose(e_model, e_decision, plan.beta)
    if not hasattr(composed, "uncertainty"):
        raise RuntimeError(f"unexpected diagnostic in positive composition: {composed}")
    L = interpretability_loss(H=H, C=0.40, O=0.18, K=0.05, U=composed.uncertainty, weights=plan.lambda_, reduction_loss=composed.reduction_loss, lambda_delta=0.10)
    I = interpretability_index(L)
    timeline.append({"step": 5, "name": "Compose explanations", "payload": {"gamma": round(gamma, 6), "L_ext": round(L, 6), "I": round(I, 6), "uncertainty": round(composed.uncertainty, 6)}})

    diagnostic = compose(e_model, e_decision.copy_with(terms={"allow", "deny"}), plan.beta)
    timeline.append({"step": 6, "name": "Trigger diagnostic state", "payload": {"diagnostic_code": getattr(diagnostic, "code", None), "reason": getattr(diagnostic, "reason", None)}})

    levels = synthesize_levels({"u_num", "u_ling", "u_time", "u_cf", "u_trace"})
    timeline.append({"step": 7, "name": "Synthesize F_ML levels for temporal/counterfactual uncertainty", "payload": {"levels": [sorted(level) for level in levels]}})

    explanation_payload = {
        "risk": risk,
        "memberships": mu,
        "selected_class": getattr(selected, "name", "FML-audit"),
        "uncertainty": round(composed.uncertainty, 6),
        "reduction_loss": round(delta, 6),
        "index": round(I, 6),
        "diagnostic_code": getattr(diagnostic, "code", None),
    }
    text_explanation = generate_human_explanation(explanation_payload, audience="doctor")
    timeline.append({"step": 8, "name": "Generate deterministic human-readable explanation", "payload": {"text": text_explanation}})

    html_path = OUT / "thesis_demo_composition_graph.html"
    try:
        save_composition_html([("model", e_model, "decision", e_decision)], plan.beta, html_path)
        graph_file = str(html_path.relative_to(ROOT))
    except Exception as exc:
        graph_file = f"not generated: {exc}"

    report = {
        "title": "FuzzyXAI Thesis Demo",
        "status": "PASS",
        "route": "metadata -> P_sit -> F* -> A_k^F -> Delta -> E_k_ext -> d_E_ext -> composition -> I(E_G) -> D_ij",
        "timeline": timeline,
        "artifacts": {
            "graph_html": graph_file,
            "json_report": "reports/thesis_demo_report.json",
            "markdown_report": "reports/thesis_demo_report.md",
        },
    }
    return report


def write_markdown(report: Dict[str, Any], path: Path) -> None:
    lines = ["# FuzzyXAI thesis demo report", "", f"Status: **{report['status']}**", "", f"Route: `{report['route']}`", "", "## Timeline", ""]
    for step in report["timeline"]:
        lines.append(f"### {step['step']}. {step['name']}")
        lines.append("")
        lines.append("```json")
        lines.append(json.dumps(step["payload"], ensure_ascii=False, indent=2))
        lines.append("```")
        lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    report = run_demo()
    json_path = OUT / "thesis_demo_report.json"
    md_path = OUT / "thesis_demo_report.md"
    json_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    write_markdown(report, md_path)
    print(json.dumps({"status": report["status"], "json": str(json_path), "markdown": str(md_path), "graph": report["artifacts"]["graph_html"]}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
