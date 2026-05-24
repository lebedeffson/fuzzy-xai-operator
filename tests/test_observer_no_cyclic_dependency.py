from __future__ import annotations

from fuzzyxai.risk import RiskAction, RiskPolicy, compute_application_risk


def test_changing_final_interpretability_does_not_change_rho():
    rho_a = compute_application_risk(0.7, 0.2, pre_interpretability=0.8, reduction_loss=0.1).rho
    final_i_changed = 0.1
    rho_b = compute_application_risk(0.7, 0.2, pre_interpretability=0.8, reduction_loss=0.1).rho
    assert final_i_changed != 0.8
    assert rho_a == rho_b


def test_changing_pre_interpretability_changes_rho():
    rho_high_i = compute_application_risk(0.7, 0.2, pre_interpretability=0.9, reduction_loss=0.1).rho
    rho_low_i = compute_application_risk(0.7, 0.2, pre_interpretability=0.3, reduction_loss=0.1).rho
    assert rho_low_i > rho_high_i


def test_policy_can_use_precomputed_application_risk():
    policy = RiskPolicy(theta_mid=0.3, theta_high=0.8)
    decision = policy.choose_from_risk(0.4, uncertainty=0.1, predicted_risk=0.6)
    assert decision.action == RiskAction.LOWER_CONFIDENCE
