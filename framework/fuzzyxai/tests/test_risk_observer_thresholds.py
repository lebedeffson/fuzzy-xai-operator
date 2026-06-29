from __future__ import annotations

from fuzzyxai import ExplainPlan
from fuzzyxai.risk import RiskAwareObserver, attach_risk_weights_to_plan, load_calibrated_risk_weights, load_thresholds


def test_load_thresholds_from_plan_metadata():
    plan = ExplainPlan(metadata={'thresholds': [0.11, 0.22, 0.55, 0.88]})
    assert load_thresholds(plan) == (0.11, 0.22, 0.55, 0.88)


def test_risk_aware_observer_threshold_decisions():
    weights = load_calibrated_risk_weights()
    plan = attach_risk_weights_to_plan(ExplainPlan(metadata={'thresholds': [0.1, 0.25, 0.5, 0.75]}), weights)
    observer = RiskAwareObserver(plan=plan)
    assert observer.decide(0.05, False) == 'accept'
    assert observer.decide(0.2, False) == 'lower_confidence'
    assert observer.decide(0.4, False) == 'request_more_data'
    assert observer.decide(0.7, False) == 'defer_to_human'
    assert observer.decide(0.2, True) == 'block'
