from __future__ import annotations

import json
from pathlib import Path

import sys
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from fuzzyxai import ExplainPlan, Rule, SystemOperator, Trace, compose
from fuzzyxai.core.trust_evaluator import (
    entropy_from_memberships,
    interpretability_index,
    interpretability_loss,
    semantic_disagreement,
)
from fuzzyxai.calibration import calibrate_beta_weights, predict_disagreement
from fuzzyxai.visual import composition_graph_dot, edge_report, save_composition_html


OUT = Path(__file__).resolve().parents[1] / "reports"
OUT.mkdir(exist_ok=True)


def build_objects():
    plan = ExplainPlan().with_reduction_weight(0.10)
    op = SystemOperator(plan)

    rules_model = [
        Rule("r_high_check", {"risk": "high"}, "additional_check"),
        Rule("r_medium_watch", {"risk": "medium"}, "watch"),
    ]
    rules_decision = [
        Rule("r_decision_high", {"risk": "high"}, "send_to_check"),
    ]

    e_model = op.explain_scalar_risk(
        0.72,
        rules_model,
        Trace("case-001-model", "v1", "2026-05-22T10:00:00", source="demo", checksum="m1"),
        model_uncertainty=0.08,
        trace_uncertainty=0.02,
    )
    e_decision = op.explain_scalar_risk(
        0.74,
        rules_decision,
        Trace("case-001-decision", "v1", "2026-05-22T10:00:01", source="demo", checksum="d1"),
        model_uncertainty=0.10,
        trace_uncertainty=0.01,
    )
    return plan, e_model, e_decision


def main():
    plan, e_model, e_decision = build_objects()

    mu = e_model.metadata["memberships"]
    H = entropy_from_memberships([mu["low"], mu["medium"], mu["high"]], epsilon=plan.epsilon)
    gamma = semantic_disagreement(e_model, e_decision, plan.beta)
    composed = compose(e_model, e_decision, plan.beta)
    assert hasattr(composed, "uncertainty"), composed

    L = interpretability_loss(
        H=H,
        C=0.40,
        O=0.18,
        K=0.05,
        U=composed.uncertainty,
        weights=plan.lambda_,
        reduction_loss=composed.reduction_loss,
        lambda_delta=0.10,
    )
    I = interpretability_index(L)

    # Proof of diagnostic behavior: no shared terms => D_ij.
    e_bad = e_decision.copy_with(terms={"allow", "deny"})
    diagnostic = compose(e_model, e_bad, plan.beta)

    # Proof of calibratability: beta can be fit from expert disagreement labels.
    feature_rows = [
        {"repr": 0.08, "rules": 0.20, "activations": 0.10, "uncertainty": 0.034, "trace": 0.05, "reduction": 0.0},
        {"repr": 0.30, "rules": 0.40, "activations": 0.30, "uncertainty": 0.10, "trace": 0.30, "reduction": 0.0},
        {"repr": 0.05, "rules": 0.00, "activations": 0.05, "uncertainty": 0.02, "trace": 0.00, "reduction": 0.0},
        {"repr": 0.55, "rules": 0.70, "activations": 0.40, "uncertainty": 0.30, "trace": 0.50, "reduction": 0.1},
    ]
    expert_scores = [0.12, 0.36, 0.04, 0.63]
    beta_cal = calibrate_beta_weights(feature_rows, expert_scores, include_reduction=True)

    edges = [("model", e_model, "decision", e_decision)]
    dot = composition_graph_dot(edges, plan.beta)
    (OUT / "chapter2_composition_graph.dot").write_text(dot, encoding="utf-8")
    try:
        save_composition_html(edges, plan.beta, OUT / "chapter2_composition_graph.html")
    except Exception as exc:
        (OUT / "chapter2_composition_graph_html_error.txt").write_text(str(exc), encoding="utf-8")

    report = {
        "chapter": 2,
        "risk": e_model.metadata["risk"],
        "memberships": mu,
        "entropy_H": round(H, 6),
        "gamma_d_E": round(gamma, 6),
        "composed_uncertainty": round(composed.uncertainty, 6),
        "loss_L": round(L, 6),
        "index_I": round(I, 6),
        "diagnostic_code": getattr(diagnostic, "code", None),
        "diagnostic_reason": getattr(diagnostic, "reason", None),
        "calibrated_beta": {k: round(v, 6) for k, v in beta_cal.items()},
        "calibration_prediction_example": round(predict_disagreement(feature_rows[0], beta_cal), 6),
        "edge_report": edge_report(edges, plan.beta),
    }
    (OUT / "chapter2_proof.json").write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")

    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
