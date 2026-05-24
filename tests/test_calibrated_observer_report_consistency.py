from __future__ import annotations

import json

import pytest

from experiments.chapter5_experiments import SCENARIOS, decision_for
from fuzzyxai import ExplainPlan
from fuzzyxai.risk import attach_risk_weights_to_plan, load_calibrated_risk_weights, policy_from_calibration, weights_from_explain_plan


def test_calibrated_policy_reproduces_chapter5_report_actions():
    report = json.loads(open('reports/chapter5/chapter5_experiments.json', encoding='utf-8').read())
    weights = load_calibrated_risk_weights()
    by_name = {sc.scenario: sc for sc in SCENARIOS}
    for row in report['tables']['scenarios_s0_s6']:
        action, rho = decision_for(by_name[row['scenario']], weights)
        assert action == row['actual_action']
        assert rho == pytest.approx(row['rho'])


def test_calibrated_weights_can_live_in_explain_plan_metadata():
    weights = load_calibrated_risk_weights()
    plan = attach_risk_weights_to_plan(ExplainPlan(), weights)
    assert weights_from_explain_plan(plan) == pytest.approx(weights)


def test_policy_from_calibration_blocks_rupture_s4_and_accepts_s0():
    policy = policy_from_calibration()
    weights = dict(policy.risk_weights)
    s4 = next(sc for sc in SCENARIOS if sc.scenario == 'S4')
    s0 = next(sc for sc in SCENARIOS if sc.scenario == 'S0')
    assert decision_for(s4, weights)[0] == 'block'
    assert decision_for(s0, weights)[0] == 'accept'
