from fuzzyxai.calibration import calibrate_beta_weights, predict_disagreement
from fuzzyxai.visual import composition_graph_dot
from fuzzyxai import ExplainPlan, Rule, SystemOperator, Trace


def test_calibration_weights_sum_to_one():
    rows = [
        {"repr": 0.1, "rules": 0.2, "activations": 0.1, "uncertainty": 0.0, "trace": 0.0, "reduction": 0.0},
        {"repr": 0.5, "rules": 0.4, "activations": 0.3, "uncertainty": 0.2, "trace": 0.1, "reduction": 0.1},
    ]
    beta = calibrate_beta_weights(rows, [0.1, 0.5], steps=20)
    assert abs(sum(beta.values()) - 1.0) < 1e-9
    assert 0 <= predict_disagreement(rows[0], beta) <= 1


def test_graph_dot_contains_edge():
    plan = ExplainPlan()
    op = SystemOperator(plan)
    rules = [Rule("r", {"risk": "high"}, "check")]
    e1 = op.explain_scalar_risk(0.72, rules, Trace("a", "v", "t"))
    e2 = op.explain_scalar_risk(0.74, rules, Trace("b", "v", "t"))
    dot = composition_graph_dot([("a", e1, "b", e2)], plan.beta)
    assert "a -> b" in dot
    assert "gamma=" in dot
