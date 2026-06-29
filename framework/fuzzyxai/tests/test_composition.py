from fuzzyxai import ExplainPlan, Rule, SystemOperator, Trace, compose

def test_composition_returns_object():
    plan = ExplainPlan()
    op = SystemOperator(plan)
    rules = [Rule('r1', {'risk':'high'}, 'check')]
    e1 = op.explain_scalar_risk(0.72, rules, Trace('a','v1','t', source='demo', checksum='1'))
    e2 = op.explain_scalar_risk(0.74, rules, Trace('b','v1','t', source='demo', checksum='2'))
    comp = compose(e1, e2, plan.beta)
    assert hasattr(comp, 'uncertainty')
    assert 0 <= comp.uncertainty <= 1

def test_composition_diagnostic_on_no_terms():
    plan = ExplainPlan()
    op = SystemOperator(plan)
    e1 = op.explain_scalar_risk(0.72, [Rule('r1', {'risk':'high'}, 'check')], Trace('a','v1','t'))
    e2 = op.explain_scalar_risk(0.74, [Rule('r2', {'risk':'high'}, 'check')], Trace('b','v1','t'))
    e2.terms = {'unrelated'}
    comp = compose(e1, e2, plan.beta)
    assert getattr(comp, 'code', '') == 'D_ij'
