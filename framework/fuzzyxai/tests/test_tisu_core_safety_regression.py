"""Regression tests for TISU core composition and pseudometric components."""
from __future__ import annotations

from fuzzyxai import ExplainPlan, Rule, SystemOperator, Trace, compose
from fuzzyxai.core.trust_evaluator import activation_distance, semantic_disagreement


def _obj(*, conclusion="check", reduction_loss=0.0):
    plan = ExplainPlan().with_reduction_weight(0.1)
    obj = SystemOperator(plan).explain_scalar_risk(
        0.72,
        [Rule("r_high", {"risk": "high"}, conclusion)],
        Trace("same", "v1", "time", source="source", checksum="checksum"),
    )
    return plan, obj.copy_with(reduction_loss=reduction_loss)


def test_conflicting_full_rule_signatures_are_not_silently_dropped():
    plan, left = _obj(conclusion="accept")
    _, right = _obj(conclusion="reject")
    result = compose(left, right, plan.beta)
    assert getattr(result, "code", None) == "D_ij"
    assert "signature" in result.reason


def test_conflicting_rule_id_within_one_object_is_refused():
    plan, left = _obj()
    left.rules.append(Rule("r_high", {"risk": "low"}, "reject"))
    _, right = _obj()
    result = compose(left, right, plan.beta)
    assert getattr(result, "code", None) == "D_ij"


def test_matching_signatures_remain_composable():
    plan, left = _obj()
    _, right = _obj()
    result = compose(left, right, plan.beta)
    assert getattr(result, "code", None) != "D_ij"
    assert len(result.rules) == 1


def test_nonzero_reduction_loss_has_zero_self_distance():
    plan, obj = _obj(reduction_loss=0.2)
    assert semantic_disagreement(obj, obj, plan.beta) == 0.0


def test_activation_union_distance_triangle_counterexample_fixed():
    _, base = _obj()
    a = base.copy_with(activations={"r3": 0.0})
    b = base.copy_with(activations={"r2": 0.0})
    c = base.copy_with(activations={"r3": 1.0})
    assert activation_distance(a, c) <= (
        activation_distance(a, b) + activation_distance(b, c)
    )
