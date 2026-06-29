from __future__ import annotations

from fuzzyxai.risk.decision_policy import RiskAction, RiskPolicy


def test_decision_policy_blocks_on_critical_rupture() -> None:
    policy = RiskPolicy()
    decision = policy.choose_from_risk(
        application_risk=0.02,
        uncertainty=0.01,
        predicted_risk=0.02,
        pre_interpretability=0.95,
        reduction_loss=0.01,
        diagnostics=[],
        chi_r=1,
        chi_r_crit=1,
        chi_auto=True,
        trace_valid=True,
    )
    assert decision.action == RiskAction.BLOCK
