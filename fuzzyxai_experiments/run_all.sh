#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"
CLEAN_LINK=0
if [ ! -e fuzzyxai_experiments ]; then
  ln -s . fuzzyxai_experiments
  CLEAN_LINK=1
fi
cleanup() {
  if [ "$CLEAN_LINK" = "1" ]; then rm -f fuzzyxai_experiments; fi
}
trap cleanup EXIT
export PYTHONPATH="$SCRIPT_DIR:$SCRIPT_DIR/..:${PYTHONPATH:-}"
mkdir -p logs reports/chapter4 reports/chapter5 tables
: > logs/run_all.log
run_step() {
  local name="$1"; shift
  echo "RUN: ${name}"
  "$@" >"logs/${name}.log" 2>&1
  echo "PASS: ${name}" | tee -a logs/run_all.log
}
run_step ch2_bc python -m fuzzyxai_experiments.experiments.ch2_bc_breast_cancer
run_step ch2_synthesis python -m fuzzyxai_experiments.experiments.ch2_synthesis_alignment
run_step ch2_critical_ruptures python -m fuzzyxai_experiments.experiments.ch2_critical_ruptures
run_step ch3_selection python -m fuzzyxai_experiments.experiments.ch3_selection_demo
run_step ch3_reduction python -m fuzzyxai_experiments.experiments.ch3_reduction_loss
run_step ch3_diagnostic_stand python -m fuzzyxai_experiments.experiments.ch3_diagnostic_stand
bash run_chapter4_5.sh
printf 'PASS: run_all\n' | tee -a logs/run_all.log
