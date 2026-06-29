from __future__ import annotations

from fuzzyxai.risk import choose_min_expected_cost


def test_choose_min_expected_cost_selects_minimum():
    action, costs = choose_min_expected_cost([0.2, 0.8], {'accept': [0, 5], 'defer': [1, 1]})
    assert action == 'defer'
    assert costs[action] == min(costs.values())


def test_selected_cost_not_greater_than_accept():
    action, costs = choose_min_expected_cost([0.1, 0.9], {'accept': [0, 10], 'block': [2, 2]})
    assert costs[action] <= costs['accept']


def test_action_changes_when_cost_matrix_changes():
    action_a, _ = choose_min_expected_cost([0.8, 0.2], {'accept': [0, 5], 'block': [2, 2]})
    action_b, _ = choose_min_expected_cost([0.8, 0.2], {'accept': [5, 0], 'block': [2, 2]})
    assert action_a != action_b
