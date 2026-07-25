from fuzzyxai.experiments.h10_c4 import PRIMARY_WEIGHTS
from fuzzyxai.repair import action_cost, select_first_valid


def test_first_valid_is_deterministic(development_scenarios, calibration) -> None:
    scenario = development_scenarios[2]
    cost = lambda action: action_cost(action, "hybrid", calibration, PRIMARY_WEIGHTS)

    first = select_first_valid(scenario.actions, scenario.obligations, cost)
    second = select_first_valid(scenario.actions, scenario.obligations, cost)

    assert first == second
    assert first.feasible
