from fuzzyxai.experiments.h10_c4 import PRIMARY_WEIGHTS, execute_strategy
from fuzzyxai.repair import action_cost, select_global_minimum_cut, select_repair_all


def test_global_cut_executes_and_fully_recertifies(
    development_scenarios, calibration
) -> None:
    scenario = next(
        item
        for item in development_scenarios
        if item.mutation_family == "shared_source_downstream"
    )
    cost = lambda action: action_cost(action, "hybrid", calibration, PRIMARY_WEIGHTS)
    repair_all = select_repair_all(scenario.actions, scenario.obligations, cost)
    global_plan = select_global_minimum_cut(
        scenario.actions, scenario.obligations, cost
    )
    result = execute_strategy(
        scenario,
        global_plan,
        calibration,
        repair_all_cost=repair_all.predicted_cost,
    )

    assert result.repair_success
    assert result.recertification_success
    assert result.final_route_status == "full_success"
    assert result.new_critical_violation_count == 0
