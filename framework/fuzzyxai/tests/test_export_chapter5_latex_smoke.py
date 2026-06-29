from __future__ import annotations

from experiments.chapter5_experiments import (
    baseline_comparison,
    breast_cancer_validation,
    calibrate_weights,
    context_logic_table,
    scenario_table,
    sensitivity,
    timing_table,
    write_outputs,
)
from experiments.export_chapter5_latex import export_latex_tables


def test_export_chapter5_latex_smoke(tmp_path):
    in_dir = tmp_path / 'chapter5'
    out_dir = tmp_path / 'latex_tables'

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
    write_outputs(in_dir, tables, calibration, sens)

    produced = export_latex_tables(in_dir, out_dir)
    assert len(produced) >= 7
