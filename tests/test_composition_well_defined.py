from __future__ import annotations

from fuzzyxai import ExplainPlan, Rule, SystemOperator, Trace, compose


def _objects():
    plan = ExplainPlan()
    op = SystemOperator(plan)
    rules = [Rule('r_high', {'risk': 'high'}, 'check')]
    e1 = op.explain_scalar_risk(0.72, rules, Trace('a', 'v1', 't', source='s', checksum='1'))
    e2 = op.explain_scalar_risk(0.74, rules, Trace('b', 'v1', 't', source='s', checksum='2'))
    return plan, e1, e2


def test_composition_is_deterministic_for_same_inputs():
    plan, e1, e2 = _objects()
    c1 = compose(e1, e2, plan.beta)
    c2 = compose(e1, e2, plan.beta)
    assert c1.metadata == c2.metadata
    assert c1.activations == c2.activations
    assert c1.terms == c2.terms


def test_composition_returns_diagnostic_when_gamma_exceeds_threshold():
    plan, e1, e2 = _objects()
    beta = dict(plan.beta)
    beta['gamma_max'] = 0.0
    result = compose(e1, e2, beta)
    assert getattr(result, 'code', '') == 'D_ij'
    assert 'gamma' in result.context


def test_composition_returns_diagnostic_for_missing_term_mapping():
    plan, e1, e2 = _objects()
    result = compose(e1, e2.copy_with(terms={'allow', 'deny'}), plan.beta)
    assert getattr(result, 'code', '') == 'D_ij'


def test_composition_is_reproducible_with_same_seedless_objects():
    plan_a, e1_a, e2_a = _objects()
    plan_b, e1_b, e2_b = _objects()
    assert compose(e1_a, e2_a, plan_a.beta).metadata['gamma'] == compose(e1_b, e2_b, plan_b.beta).metadata['gamma']
