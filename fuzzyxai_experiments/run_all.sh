#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
export PYTHONPATH=.
mkdir -p fuzzyxai_experiments/logs fuzzyxai_experiments/reports/chapter4 fuzzyxai_experiments/reports/chapter5 fuzzyxai_experiments/tables
run_step() {
  local name="$1"; shift
  echo "RUN: ${name}"
  "$@" >"fuzzyxai_experiments/logs/${name}.log" 2>&1
  echo "PASS: ${name}" | tee -a fuzzyxai_experiments/logs/run_all.log
}
: > fuzzyxai_experiments/logs/run_all.log
run_step ch2_bc python -m fuzzyxai_experiments.experiments.ch2_bc_breast_cancer
run_step ch2_synthesis python -m fuzzyxai_experiments.experiments.ch2_synthesis_alignment
run_step ch2_critical_ruptures python -m fuzzyxai_experiments.experiments.ch2_critical_ruptures
run_step ch3_selection python -m fuzzyxai_experiments.experiments.ch3_selection_demo
run_step ch3_reduction python -m fuzzyxai_experiments.experiments.ch3_reduction_loss
run_step ch3_diagnostic_stand python -m fuzzyxai_experiments.experiments.ch3_diagnostic_stand
run_step ch4_integration python -m fuzzyxai_experiments.experiments.ch4_integration_metrics
run_step ch5_scenario_runs python -m fuzzyxai_experiments.experiments.ch5_scenario_runs
run_step ch5_hybrid python -m fuzzyxai_experiments.experiments.ch5_hybrid
run_step ch5_gis python -m fuzzyxai_experiments.experiments.ch5_gis
run_step ch5_beacon python -m fuzzyxai_experiments.experiments.ch5_beacon
run_step ch5_gd_anfis_shap python -m fuzzyxai_experiments.experiments.ch5_gd_anfis_shap
cp fuzzyxai_experiments/reports/ch4_integration.json fuzzyxai_experiments/reports/chapter4/ch4_integration.json
cp fuzzyxai_experiments/reports/ch5_scenario_runs.json fuzzyxai_experiments/reports/chapter5/ch5_scenario_runs.json
cp fuzzyxai_experiments/reports/ch5_hybrid.json fuzzyxai_experiments/reports/chapter5/ch5_hybrid.json
cp fuzzyxai_experiments/reports/ch5_gis.json fuzzyxai_experiments/reports/chapter5/ch5_gis.json
cp fuzzyxai_experiments/reports/ch5_beacon.json fuzzyxai_experiments/reports/chapter5/ch5_beacon.json
cp fuzzyxai_experiments/reports/ch5_gd_anfis_shap.json fuzzyxai_experiments/reports/chapter5/ch5_gd_anfis_shap.json
python fuzzyxai_experiments/generate_tables.py > fuzzyxai_experiments/tables/generated_tables.tex
cp fuzzyxai_experiments/tables/generated_tables.tex fuzzyxai_experiments/reports/generated_tables.tex
cp fuzzyxai_experiments/reports/chapter5/*.csv fuzzyxai_experiments/tables/ 2>/dev/null || true
cp fuzzyxai_experiments/reports/chapter5/*.json fuzzyxai_experiments/tables/ 2>/dev/null || true
cp fuzzyxai_experiments/reports/chapter4/*.json fuzzyxai_experiments/tables/ 2>/dev/null || true
python fuzzyxai_experiments/scripts/build_manifest.py | tee fuzzyxai_experiments/logs/manifest.log
python fuzzyxai_experiments/compare_reports.py | tee fuzzyxai_experiments/logs/compare_reports.log
printf 'PASS: run_all\n' | tee -a fuzzyxai_experiments/logs/run_all.log
