"""P16 section 17: MembershipPolicy — a single, disclosed source of fuzzy
membership functions (variable, universe, terms, parameters, origin,
version, calibration reference) instead of untraceable numbers. Registered
policies are always disclosed alongside any membership evidence; absent a
registration, nothing is fabricated.
"""

from __future__ import annotations

import pytest
from fuzzyxai import FuzzyXAI
from fuzzyxai.core.explain_plan import ExplainPlan, MembershipPolicy, MembershipTerm
from sklearn.datasets import load_breast_cancer
from sklearn.linear_model import LogisticRegression


def _policy() -> MembershipPolicy:
    return MembershipPolicy(
        variable="risk",
        universe=(0.0, 1.0),
        terms=(
            MembershipTerm("low", "triangular", (0.0, 0.0, 0.3)),
            MembershipTerm("medium", "triangular", (0.2, 0.5, 0.8)),
            MembershipTerm("high", "triangular", (0.6, 1.0, 1.0)),
        ),
        origin="calibrated",
        version="v1",
        calibration_reference="chapter5/chapter5_experiments.json",
    )


def test_membership_policy_rejects_unknown_origin() -> None:
    with pytest.raises(ValueError, match="origin"):
        MembershipPolicy(variable="risk", universe=(0.0, 1.0), terms=(MembershipTerm("low", "triangular", (0.0, 0.0, 0.3)),), origin="guessed", version="v1")


def test_membership_policy_requires_at_least_one_term() -> None:
    with pytest.raises(ValueError, match="term"):
        MembershipPolicy(variable="risk", universe=(0.0, 1.0), terms=(), origin="expert", version="v1")


def test_membership_policy_evaluates_shoulders_and_regular_triangle() -> None:
    policy = _policy()
    assert policy.evaluate(0.0)["low"] == 1.0
    assert policy.evaluate(1.0)["high"] == 1.0
    assert policy.evaluate(0.5)["medium"] == 1.0
    assert policy.evaluate(-0.1)["low"] == 0.0
    assert policy.evaluate(1.1)["high"] == 0.0


def test_explain_plan_round_trips_membership_policies() -> None:
    plan = ExplainPlan(membership_policies={"risk": _policy()})
    restored = ExplainPlan.from_dict(plan.to_dict())
    assert restored.membership_policies["risk"].origin == "calibrated"
    assert restored.membership_policies["risk"].terms[1].parameters == (0.2, 0.5, 0.8)
    assert restored.membership_policies["risk"].calibration_reference == "chapter5/chapter5_experiments.json"


def test_registered_policy_is_disclosed_alongside_membership_evidence() -> None:
    X, y = load_breast_cancer(return_X_y=True)
    model = LogisticRegression(max_iter=3000).fit(X, y)
    plan = ExplainPlan(membership_policies={"risk": _policy()})
    result = FuzzyXAI.wrap(model, explain_plan=plan).explain_one(X[0], object_id="p0", evidence={"memberships": {"medium": 0.34, "high": 0.71}})
    policies = result.view_model.fuzzy["membership_policies"]
    assert policies["risk"]["origin"] == "calibrated"
    assert policies["risk"]["terms"][0]["parameters"] == [0.0, 0.0, 0.3]


def test_no_policy_registered_discloses_empty_not_fabricated() -> None:
    X, y = load_breast_cancer(return_X_y=True)
    model = LogisticRegression(max_iter=3000).fit(X, y)
    result = FuzzyXAI.wrap(model).explain_one(X[0], object_id="p0", evidence={"memberships": {"medium": 0.34}})
    assert result.view_model.fuzzy["membership_policies"] == {}
