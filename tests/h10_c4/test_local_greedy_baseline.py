from fuzzyxai.experiments.h10_c4 import PRIMARY_WEIGHTS
from fuzzyxai.repair import action_cost, select_local_greedy


def test_local_greedy_covers_registered_obligations(
    development_scenarios, calibration
) -> None:
    scenario = next(
        item
        for item in development_scenarios
        if item.mutation_family == "equal_size_different_cost"
    )
    cost = lambda action: action_cost(action, "hybrid", calibration, PRIMARY_WEIGHTS)
    plan = select_local_greedy(scenario.actions, scenario.obligations, cost)

    assert scenario.obligations.issubset(plan.covered_obligations)
