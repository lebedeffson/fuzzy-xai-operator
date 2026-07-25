from fuzzyxai.experiments.h10_c4 import _family_results, run_scenarios
from fuzzyxai.repair import action_cost, select_global_minimum_cut


def test_family_results_keep_all_six_independent_families(
    development_scenarios, calibration
) -> None:
    rows = _family_results(run_scenarios(development_scenarios, calibration))

    assert len({row["pipeline_family"] for row in rows}) == 6
    assert all(
        any(
            row["pipeline_family"] == family
            and row["strategy"] == "O_GLOBAL_vs_B_ALL"
            for row in rows
        )
        for family in {row["pipeline_family"] for row in rows}
    )


def test_each_registered_cost_model_selects_a_recertifiable_cut(
    development_scenarios, calibration
) -> None:
    scenario = development_scenarios[4]
    for model in ("uniform", "runtime", "dependency_weighted", "hybrid"):
        cost = lambda action, selected=model: action_cost(
            action,
            selected,
            calibration,
        )
        plan = select_global_minimum_cut(
            scenario.actions,
            scenario.obligations,
            cost,
        )
        assert plan.feasible
        assert scenario.obligations.issubset(plan.covered_obligations)
