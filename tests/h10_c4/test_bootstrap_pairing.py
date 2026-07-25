from fuzzyxai.experiments.h10_c4 import analyze, run_scenarios


def test_all_primary_contrasts_share_one_bootstrap_index_stream(
    development_scenarios, calibration
) -> None:
    results = run_scenarios(development_scenarios, calibration)
    comparisons, _, _ = analyze(results)

    assert len(
        {row["bootstrap_index_stream_sha256"] for row in comparisons}
    ) == 1


def test_pipeline_families_have_distinct_serialized_graphs(
    development_scenarios,
) -> None:
    first_by_family = {}
    for scenario in development_scenarios:
        first_by_family.setdefault(scenario.pipeline_family, scenario)

    edge_counts = {
        family: len(scenario.valid_graph.edges)
        for family, scenario in first_by_family.items()
    }
    assert len(set(edge_counts.values())) >= 3
    assert len(
        {scenario.route_graph_hash for scenario in first_by_family.values()}
    ) == 6
