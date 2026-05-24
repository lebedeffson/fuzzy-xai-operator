from __future__ import annotations

from experiments.chapter2_breast_cancer_demo import run as run_ch2, write_reports as write_ch2_reports
from experiments.chapter5_experiments import (
    baseline_comparison,
    calibrate_weights,
    scenario_table,
    sensitivity,
    timing_table,
    write_outputs,
)
from experiments.generate_figures import generate_figures
from experiments.chapter5_experiments import breast_cancer_validation, context_logic_table


def test_generate_figures_smoke(tmp_path):
    chapter2_dir = tmp_path / 'chapter2'
    chapter5_dir = tmp_path / 'chapter5'
    figures_dir = tmp_path / 'figures'

    df2, summary2 = run_ch2(seed=1, max_cases=24)
    write_ch2_reports(df2, summary2, chapter2_dir)

    calibration = calibrate_weights()
    weights = calibration['weights']
    tables = {
        'scenarios_s0_s6': scenario_table(weights),
        'baseline_comparison': baseline_comparison(weights, 5, 1),
        'timing_complexity': timing_table(weights, 20, 1),
        'breast_cancer_validation': breast_cancer_validation(weights, 1),
        'context_logic': context_logic_table(),
    }
    sens = sensitivity(weights, 5, 1)
    write_outputs(chapter5_dir, tables, calibration, sens)

    produced = generate_figures(chapter2_dir, chapter5_dir, figures_dir)
    assert 'i_pre_html' in produced
    assert 'wR_html' in produced
    assert 'theta_html' in produced
    assert 'baseline_html' in produced
