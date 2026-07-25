from fuzzyxai.experiments.h10_c4 import PRIMARY_WEIGHTS, execute_strategy
from fuzzyxai.repair import StrategyPlan, action_cost, select_repair_all


def test_rollback_cost_is_counted_as_an_executed_operation(
    development_scenarios, calibration
) -> None:
    scenario = next(
        item
        for item in development_scenarios
        if item.mutation_family == "rollback_required"
    )
    unsafe = next(
        action for action in scenario.actions if action.creates_critical_violation
    )
    cost = lambda action: action_cost(action, "hybrid", calibration, PRIMARY_WEIGHTS)
    repair_all = select_repair_all(scenario.actions, scenario.obligations, cost)
    result = execute_strategy(
        scenario,
        StrategyPlan(
            "TEST_ROLLBACK",
            (unsafe.action_id,),
            cost(unsafe),
            tuple(sorted(unsafe.covers)),
            False,
        ),
        calibration,
        repair_all_cost=repair_all.predicted_cost,
    )

    assert result.rollback_count == 1
    assert result.executed_cost == 2 * cost(unsafe)
