from fuzzyxai.risk import RiskAction, RiskPolicy


def test_diagnostic_blocks_automatic_decision():
    decision = RiskPolicy().choose(0.2, 0.1, 0.9, 0.0, diagnostics=['D_ij'])
    assert decision.action == RiskAction.BLOCK
    assert decision.corrected_confidence == 0.0


def test_high_risk_high_uncertainty_defers():
    policy = RiskPolicy(theta_mid=0.30, theta_high=0.55)
    decision = policy.choose(0.95, 0.85, 0.5, 0.2)
    assert decision.action == RiskAction.DEFER_TO_HUMAN


def test_medium_risk_can_lower_confidence():
    policy = RiskPolicy(theta_mid=0.20, theta_high=0.80)
    decision = policy.choose(0.6, 0.2, 0.8, 0.0)
    assert decision.action == RiskAction.LOWER_CONFIDENCE
    assert 0 <= decision.corrected_confidence <= 1
