from __future__ import annotations

import pytest

from fuzzyxai.risk import (
    choose_min_expected_cost,
    compute_application_risk,
    expected_action_costs,
    normalize_risk_weights,
)
from fuzzyxai.risk.decision_policy import RiskPolicy


def test_application_risk_matches_observer_formula():
    breakdown = compute_application_risk(
        predicted_risk=0.8,
        uncertainty=0.4,
        interpretability=0.7,
        reduction_loss=0.2,
        diagnostics=['D_RD'],
    )
    assert breakdown.rho == pytest.approx(0.53)
    assert breakdown.interpretability_gap == pytest.approx(0.3)
    assert breakdown.diagnostic == 1.0


def test_risk_policy_delegates_to_application_risk():
    policy = RiskPolicy(block_on_diagnostic=False)
    direct = compute_application_risk(0.8, 0.4, 0.7, 0.2, ['D_RD']).rho
    assert policy.risk_score(0.8, 0.4, 0.7, 0.2, ['D_RD']) == pytest.approx(direct)


def test_risk_weights_are_normalized_to_simplex():
    weights = normalize_risk_weights({'predicted_risk': 2, 'uncertainty': 2, 'diagnostic': 0})
    assert sum(weights.values()) == pytest.approx(1.0)
    assert all(v >= 0 for v in weights.values())


def test_expected_cost_policy_never_worse_than_accept_by_own_cost():
    action, costs = choose_min_expected_cost(
        [0.2, 0.8],
        {
            'accept': [0.0, 5.0],
            'defer_to_human': [0.5, 0.5],
            'block': [1.0, 1.0],
        },
    )
    assert action == 'defer_to_human'
    assert costs[action] <= costs['accept']
    assert expected_action_costs([0.2, 0.8], {'accept': [0.0, 5.0]})['accept'] == pytest.approx(4.0)
