from fuzzyxai.experiments.h10_c4 import calibrate_runtime, run_scenarios


def test_global_strategy_has_full_postcondition_based_recertification(
    development_scenarios,
) -> None:
    calibration = calibrate_runtime(development_scenarios)
    results = run_scenarios(development_scenarios[:3], calibration)
    global_rows = [row for row in results if row.strategy == "O_GLOBAL"]

    assert global_rows
    assert all(row.postconditions_pass for row in global_rows)
    assert all(row.verifier_consistency == "PASS" for row in global_rows)
    assert all(row.repair_success for row in global_rows)
