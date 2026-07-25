from fuzzyxai.experiments.h10_c4 import PRIMARY_WEIGHTS, execute_strategy
from fuzzyxai.repair import StrategyPlan, action_cost, select_repair_all


def test_secondary_critical_violation_triggers_verified_rollback(
    development_scenarios, calibration
) -> None:
    scenario = next(
        item
        for item in development_scenarios
        if item.mutation_family == "secondary_critical_violation"
    )
    unsafe = next(
        action for action in scenario.actions if action.creates_critical_violation
    )
    cost = lambda action: action_cost(action, "hybrid", calibration, PRIMARY_WEIGHTS)
    repair_all = select_repair_all(scenario.actions, scenario.obligations, cost)
    plan = StrategyPlan(
        "TEST_UNSAFE",
        (unsafe.action_id,),
        cost(unsafe),
        tuple(sorted(unsafe.covers)),
        False,
    )
    result = execute_strategy(
        scenario,
        plan,
        calibration,
        repair_all_cost=repair_all.predicted_cost,
    )

    assert result.rollback_count == 1
    assert not result.repair_success
    assert result.after_trace_sha256 == result.before_trace_sha256
