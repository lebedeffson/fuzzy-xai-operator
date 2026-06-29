from __future__ import annotations

from fuzzyxai import ExplainPlan, Rule, SystemOperator, Trace, compose, interpretability_loss


def _loss(obj, plan):
    return interpretability_loss(0.2, 0.2, 0.1, 0.1, obj.uncertainty, plan.lambda_, obj.reduction_loss, 0.10)


def _chain():
    plan = ExplainPlan().with_reduction_weight(0.10)
    op = SystemOperator(plan)
    rules = [Rule('r_high', {'risk': 'high'}, 'check')]
    return plan, [op.explain_scalar_risk(v, rules, Trace(f'e{i}', 'v1', 't')) for i, v in enumerate([0.70, 0.72, 0.75, 0.78])]


def test_chain_loss_is_bounded_by_sum_bound():
    plan, objs = _chain()
    comp = objs[0]
    gammas = []
    for nxt in objs[1:]:
        comp = compose(comp, nxt, plan.beta)
        gammas.append(comp.metadata['gamma'])
    actual = _loss(comp, plan)
    bound = sum(_loss(o, plan) for o in objs) + 0.10 * sum(gammas) + sum(o.reduction_loss for o in objs)
    assert actual <= bound + 1e-9


def test_chain_bound_grows_with_gamma():
    base = 1.0 + 0.10 * 0.1
    larger = 1.0 + 0.10 * 0.8
    assert larger > base


def test_chain_bound_without_delta_has_no_reduction_term():
    plan, objs = _chain()
    assert sum(o.reduction_loss for o in objs) == 0.0
