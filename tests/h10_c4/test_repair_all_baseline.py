from fuzzyxai.experiments.h10_c4 import PRIMARY_WEIGHTS
from fuzzyxai.repair import action_cost, select_repair_all


def test_repair_all_selects_each_direct_violated_component(
    development_scenarios, calibration
) -> None:
    scenario = next(
        item
        for item in development_scenarios
        if item.mutation_family == "independent_multiple"
    )
    cost = lambda action: action_cost(action, "hybrid", calibration, PRIMARY_WEIGHTS)
    plan = select_repair_all(scenario.actions, scenario.obligations, cost)

    assert plan.feasible
    assert len(plan.action_ids) == len(scenario.obligations)
    assert all(item.startswith("direct:") for item in plan.action_ids)
