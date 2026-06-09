#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
export PYTHONPATH=.
python -m fuzzyxai_experiments.experiments.ch2_bc_breast_cancer >/tmp/fx_ch2_bc.log
python -m fuzzyxai_experiments.experiments.ch2_synthesis_alignment >/tmp/fx_ch2_synth.log
python -m fuzzyxai_experiments.experiments.ch2_critical_ruptures >/tmp/fx_ch2_crit.log
python -m fuzzyxai_experiments.experiments.ch3_selection_demo >/tmp/fx_ch3_select.log
python -m fuzzyxai_experiments.experiments.ch3_reduction_loss >/tmp/fx_ch3_reduction.log
python -m fuzzyxai_experiments.experiments.ch3_diagnostic_stand >/tmp/fx_ch3_diag.log
python -m fuzzyxai_experiments.experiments.ch4_integration_metrics >/tmp/fx_ch4.log
python -m fuzzyxai_experiments.experiments.ch5_scenario_runs >/tmp/fx_ch5_scenarios.log
python -m fuzzyxai_experiments.experiments.ch5_hybrid >/tmp/fx_ch5_hybrid.log
python -m fuzzyxai_experiments.experiments.ch5_gis >/tmp/fx_ch5_gis.log
python -m fuzzyxai_experiments.experiments.ch5_beacon >/tmp/fx_ch5_beacon.log
python fuzzyxai_experiments/generate_tables.py > fuzzyxai_experiments/reports/generated_tables.tex
echo "Generated reports:"
python - <<'PY'
from pathlib import Path
for p in sorted(Path('fuzzyxai_experiments/reports').glob('ch*.json')):
    print('-', p)
PY
