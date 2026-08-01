from fuzzyxai.experiments.h10_c4 import PRIMARY_WEIGHTS
from fuzzyxai.repair import action_cost, select_global_minimum_cut


def test_all_equal_cost_optimal_cuts_are_preserved(
    development_scenarios, calibration
) -> None:
    scenario = next(
        item
        for item in development_scenarios
        if item.mutation_family == "multiple_equal_optima"
    )
    cost = lambda action: action_cost(action, "hybrid", calibration, PRIMARY_WEIGHTS)
    plan = select_global_minimum_cut(
        scenario.actions,
        scenario.obligations,
        cost,
    )

    assert len(plan.equivalent_optimal_plans) >= 2
    assert plan.action_ids in plan.equivalent_optimal_plans
