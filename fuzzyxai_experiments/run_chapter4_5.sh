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
LOG="logs/run_chapter4_5.log"
: > "$LOG"
log() { echo "$1" | tee -a "$LOG"; }
run_step() {
  local name="$1"; shift
  log "RUN: ${name}"
  "$@" >"logs/${name}.log" 2>&1
  log "PASS: ${name}"
}
python - <<'PY' > logs/ch4_registry_check.log 2>&1
import json
from pathlib import Path
modules = json.loads(Path('registry/modules.json').read_text(encoding='utf-8'))['modules']
required = {'hybrid_xiris', 'beacon_xai', 'gis_integro', 'gd_anfis_shap'}
ids = {m['registry_id'] for m in modules}
assert required <= ids, ids
for m in modules:
    assert {'registry_id', 'status', 'adapter', 'input_artifact', 'output_report', 'claim_scope'} <= set(m), m
PY
log "PASS: ch4_registry_check"
run_step hybrid_xiris python -m fuzzyxai_experiments.experiments.ch5_hybrid
run_step beacon_xai python -m fuzzyxai_experiments.experiments.ch5_beacon
run_step gis_integro python -m fuzzyxai_experiments.experiments.ch5_gis
run_step gd_anfis_shap python -m fuzzyxai_experiments.experiments.ch5_gd_anfis_shap
run_step ch5_scenario_runs python -m fuzzyxai_experiments.experiments.ch5_scenario_runs
run_step ch4_integration python -m fuzzyxai_experiments.experiments.ch4_integration_metrics
cp reports/ch4_integration.json reports/chapter4/ch4_integration.json
cp reports/ch5_hybrid.json reports/chapter5/ch5_hybrid.json
cp reports/ch5_beacon.json reports/chapter5/ch5_beacon.json
cp reports/ch5_gis.json reports/chapter5/ch5_gis.json
cp reports/ch5_gd_anfis_shap.json reports/chapter5/ch5_gd_anfis_shap.json
cp reports/ch5_scenario_runs.json reports/chapter5/ch5_scenario_runs.json
python generate_tables.py > tables/generated_tables.tex
cp tables/generated_tables.tex reports/generated_tables.tex
cp reports/chapter5/*.csv tables/ 2>/dev/null || true
cp reports/chapter5/*.json tables/ 2>/dev/null || true
cp reports/chapter4/*.json tables/ 2>/dev/null || true
python extract_tables.py > logs/extract_tables.log 2>&1
python scripts/build_gui_screenshots.py > logs/export_gui_screenshots.log 2>&1
python scripts/build_manifest.py > logs/manifest.log 2>&1
cat logs/manifest.log | tee -a "$LOG"
log "PASS: ch4_evidence_manifest"
python compare_reports.py > logs/compare_reports.log 2>&1
cat logs/compare_reports.log | tee -a "$LOG"
sha256sum -c checksums.sha256 > logs/checksums.log 2>&1
printf 'Docker daemon note: run `docker run --rm fuzzyxai/evidence:chapter4-5 bash run_chapter4_5.sh` after building image. Local daemon may be unavailable.\n' > logs/docker_run.log
log "PASS: chapter4_5_complete"
exit 0
