from fuzzyxai.risk.metrics import accepted_accuracy, coverage, expected_cost_after, expected_cost_before, risk_reduction


def test_accepted_accuracy_only_uses_accept_action():
    y_true = [1, 0, 1]
    y_pred = [1, 1, 0]
    actions = ['accept', 'defer_to_human', 'block']
    assert accepted_accuracy(y_true, y_pred, actions) == 1.0
    assert coverage(actions) == 1 / 3


def test_expected_cost_after_uses_safe_action_costs():
    y_true = [1, 1]
    y_pred = [0, 1]
    actions = ['accept', 'defer_to_human']
    matrix = [[0.0, 1.0], [5.0, 0.0]]
    before = expected_cost_before(y_true, y_pred, matrix)
    after = expected_cost_after(y_true, y_pred, actions, matrix, defer_cost=0.1)
    assert before == 2.5
    assert after == 2.55
    assert risk_reduction(before, after) < 0
