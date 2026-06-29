from __future__ import annotations

from experiments.chapter5_experiments import baseline_comparison, calibrate_weights, context_logic_table, scenario_table, timing_table


def test_chapter5_experiments_smoke():
    calibration = calibrate_weights()
    weights = calibration['weights']
    assert abs(sum(weights.values()) - 1.0) < 1e-9
    assert calibration['accuracy'] >= 0.5
    assert len(scenario_table(weights)) == 7
    assert set(baseline_comparison(weights, 2, 1)['mode']) == {'risk_threshold', 'fuzzy_operator', 'observer_no_topos', 'full_fuzzyxai'}
    assert (timing_table(weights, 5, 1)['mean_ms'] >= 0).all()
    assert 'AutoAccept' in context_logic_table().columns
