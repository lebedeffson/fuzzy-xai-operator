from __future__ import annotations

from fuzzyxai import ExplainPlan, Rule, SystemOperator, Trace
from fuzzyxai.core.trust_evaluator import semantic_disagreement


def _obj(risk: float, uncertainty: float = 0.05):
    plan = ExplainPlan()
    op = SystemOperator(plan)
    rules = [Rule('r_high', {'risk': 'high'}, 'check'), Rule('r_medium', {'risk': 'medium'}, 'watch')]
    obj = op.explain_scalar_risk(risk, rules, Trace('same', 'v1', 't', source='src', checksum='same'), model_uncertainty=uncertainty)
    return plan, obj


def test_d_e_zero_on_diagonal():
    plan, a = _obj(0.7)
    assert semantic_disagreement(a, a, plan.beta) == 0.0


def test_d_e_symmetric_on_aligned_objects():
    plan, a = _obj(0.7)
    _, b = _obj(0.75)
    assert semantic_disagreement(a, b, plan.beta) == semantic_disagreement(b, a, plan.beta)


def test_d_e_triangle_on_aligned_objects():
    plan, a = _obj(0.65)
    _, b = _obj(0.70)
    _, c = _obj(0.78)
    ab = semantic_disagreement(a, b, plan.beta)
    bc = semantic_disagreement(b, c, plan.beta)
    ac = semantic_disagreement(a, c, plan.beta)
    assert ac <= ab + bc + 1e-9
