from __future__ import annotations

import json
from pathlib import Path

import sys
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from fuzzyxai import (
    ExplainPlan,
    HesitantFS,
    IntervalFS,
    MultiLevelFS,
    NeutrosophicFS,
    Rule,
    SystemOperator,
    Trace,
    build_profile,
    compose,
    reduce_to_f0,
    select_minimal_sufficient,
)
from fuzzyxai.core.trust_evaluator import semantic_disagreement
from fuzzyxai.selection.compatibility import synthesize_levels
from fuzzyxai.selection.pareto_selector import Candidate


OUT = Path(__file__).resolve().parents[1] / "reports"
OUT.mkdir(exist_ok=True)


def make_candidates():
    return [
        Candidate("F0", {"u_num", "u_ling"}, 0.10, 0.03, 0.42),
        Candidate("FI", {"u_num", "u_ling", "u_int"}, 0.22, 0.05, 0.30),
        Candidate("FH", {"u_num", "u_ling", "u_exp"}, 0.30, 0.81, 0.25),
        Candidate("FNsrc", {"u_num", "u_ling", "u_if", "u_conf", "u_trace"}, 0.42, 0.09, 0.18),
        Candidate("FML-user", {"u_num", "u_ling", "u_int", "u_exp", "u_conf", "u_multi"}, 0.60, 0.99, 0.08),
        Candidate("FML-audit", {"u_num", "u_ling", "u_int", "u_exp", "u_conf", "u_trace", "u_multi"}, 0.68, 1.00, 0.04),
    ]


def main():
    metadata = {
        "has_intervals": True,
        "num_experts": 2,
        "source_conflict": True,
        "requires_audit": True,
        "multi_level": True,
    }
    profile = build_profile(metadata)
    selected = select_minimal_sufficient(profile, make_candidates(), mode="audit")
    assert getattr(selected, "name", None) == "FML-audit"

    interval = IntervalFS(lambda x: 0.68, lambda x: 0.76, policy="mid")
    hesitant = HesitantFS(lambda x: [0.61, 0.78])
    neutro = NeutrosophicFS(lambda x: 0.78, lambda x: 0.20, lambda x: 0.64, source_t="risk_model", source_f="expert_panel")
    multi = MultiLevelFS(
        [interval, hesitant, neutro],
        gamma={("interval", "hesitant", "same_case"), ("neutrosophic", "source", "conflict")},
        weights=[0.25, 0.25, 0.50],
    )

    red_interval, d_interval = interval.reduce_to_f0()
    red_hesitant, d_hesitant = hesitant.reduce_to_f0()
    red_neutro, d_neutro = neutro.reduce_to_f0()
    red_multi, d_multi = multi.reduce_to_f0()

    # Link to chapter 2: insert A_k^F into E_k^ext and compute d_E^ext.
    plan = ExplainPlan().with_reduction_weight(0.10)
    op = SystemOperator(plan)
    rules = [Rule("r_high_check", {"risk": "high"}, "additional_check")]
    e_ext = op.explain_scalar_risk(0.72, rules, Trace("case-001-ext", "v1", "2026-05-22T10:00:00", source="demo", checksum="ext"))
    e_ext.representation = multi
    e_ext.reduction_loss = d_multi

    e_simple = op.explain_scalar_risk(0.74, rules, Trace("case-001-simple", "v1", "2026-05-22T10:00:01", source="demo", checksum="simple"))
    d_ext = semantic_disagreement(e_ext, e_simple, plan.beta)
    composed = compose(e_ext, e_simple, plan.beta)

    # Diagnostic choice proof for an unseen required type.
    profile_with_unknown = set(profile) | {"u_ethical"}
    diag = select_minimal_sufficient(profile_with_unknown, make_candidates(), mode="audit")

    # Proof of u_time/u_cf placement in multilevel synthesis.
    levels_for_temporal_cf = synthesize_levels({"u_num", "u_ling", "u_time", "u_cf", "u_trace"})

    report = {
        "chapter": 3,
        "profile": sorted(profile),
        "selected_class": selected.name,
        "candidate_table": [c.__dict__ | {"coverage": sorted(c.coverage)} for c in make_candidates()],
        "reductions": {
            "IntervalFS_to_F0_delta": round(d_interval, 6),
            "HesitantFS_to_F0_delta": round(d_hesitant, 6),
            "NeutrosophicFS_to_F0_delta": round(d_neutro, 6),
            "MultiLevelFS_to_F0_delta": round(d_multi, 6),
        },
        "extended_d_E_with_delta": round(d_ext, 6),
        "composition_result": "ExplanationObject" if hasattr(composed, "uncertainty") else getattr(composed, "code", "unknown"),
        "diagnostic_choice_code": getattr(diag, "code", None),
        "diagnostic_choice_context": getattr(diag, "context", None),
        "u_time_u_cf_multilevel_levels": [sorted(level) for level in levels_for_temporal_cf],
    }
    (OUT / "chapter3_proof.json").write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
