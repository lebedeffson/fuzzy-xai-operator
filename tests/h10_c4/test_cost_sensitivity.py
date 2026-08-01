from fuzzyxai.repair import run_cost_sensitivity


def test_registered_cost_grid_has_48_configurations(
    development_scenarios, calibration
) -> None:
    scenario = development_scenarios[4]
    rows = run_cost_sensitivity(
        scenario.actions,
        scenario.obligations,
        calibration,
    )

    assert len(rows) == 48
    assert all(row.repair_success for row in rows)
