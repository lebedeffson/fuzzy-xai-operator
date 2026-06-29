from fuzzyxai.core.trust_evaluator import entropy_from_memberships, interpretability_index

def test_entropy_reasonable_for_risk_example():
    h = entropy_from_memberships([0.0, 0.12, 0.88], 0.001)
    assert 0.30 < h < 0.40

def test_index_decreases_with_loss():
    assert interpretability_index(0.2) > interpretability_index(0.8)
