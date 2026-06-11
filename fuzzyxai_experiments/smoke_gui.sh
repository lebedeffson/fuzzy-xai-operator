#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"
PORT="${PORT:-8501}"
LOG="logs/smoke_gui.log"
mkdir -p logs
: > "$LOG"
PORT="$PORT" bash start_gui.sh > logs/gui_server.log 2>&1 &
GUI_PID=$!
cleanup() {
  kill "$GUI_PID" >/dev/null 2>&1 || true
  wait "$GUI_PID" >/dev/null 2>&1 || true
}
trap cleanup EXIT
python - <<'PY' "$PORT" | tee -a "$LOG"
import json
import sys
import time
import urllib.error
import urllib.request

port = sys.argv[1]
base = f'http://127.0.0.1:{port}'

def request(path, method='GET'):
    req = urllib.request.Request(base + path, method=method)
    return urllib.request.urlopen(req, timeout=10).read()

last = None
for _ in range(30):
    try:
        html = request('/').decode('utf-8')
        if 'FuzzyXAI Studio' in html:
            break
    except Exception as exc:  # noqa: BLE001
        last = exc
        time.sleep(0.2)
else:
    raise SystemExit(f'FAIL: gui_home {last}')
print('PASS: gui_home')
scenarios = json.loads(request('/api/scenarios').decode('utf-8'))
assert scenarios['evidence']['status'] in {'PASS', 'FAIL', 'NOT RUN'}
assert len(scenarios['scenarios']) == 4
print('PASS: gui_api_scenarios')
checksums = json.loads(request('/api/checksums').decode('utf-8'))
assert checksums['status'] == 'PASS', checksums
print('PASS: gui_api_checksums')
registry = json.loads(request('/api/registry').decode('utf-8'))
assert len(registry['modules']) == 4
assert json.loads(request('/api/evidence/status').decode('utf-8'))['status'] == 'PASS'
assert len(json.loads(request('/api/evidence/files').decode('utf-8'))['files']) == 4
assert json.loads(request('/api/reports/hybrid_xiris').decode('utf-8'))['total_objects'] == 1000
print('PASS: gui_extra_endpoints')
file_text = request('/files/reports/chapter5/hybrid_xiris_summary.json').decode('utf-8')
assert 'baseline_missed' in file_text
print('PASS: gui_files')
export = json.loads(request('/api/gui/export-screenshots', method='POST').decode('utf-8'))
assert export['ok'], export
print('PASS: gui_export_screenshots')
PY
