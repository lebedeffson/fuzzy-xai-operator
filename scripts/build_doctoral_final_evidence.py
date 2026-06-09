from __future__ import annotations

import hashlib
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
COMMANDS = [
    ['make', 'reproduce-chapter2'],
    ['make', 'calibrate-chapter2'],
    ['make', 'benchmark-equal-raw-structure'],
    ['make', 'reproduce-critical-ruptures'],
    ['make', 'ecosystem-evidence'],
    ['make', 'dissertation-artifacts'],
]
KEY_FILES = [
    'reports/chapter2/sample_113_report.json',
    'reports/chapter2/calibration_report.json',
    'reports/chapter2/equal_raw_structure_report.json',
    'reports/chapter2/alignment_synthesis_report.json',
    'reports/chapter3/synthetic_ruptures_summary.json',
    'reports/chapter3/synthetic_ruptures_results.csv',
    'reports/chapter4/ecosystem_evidence.json',
    'reports/chapter4/integration_effort_summary.json',
    'reports/chapter5/scenario_run_summary.csv',
    'reports/chapter5/hybrid_xiris_blocking_case.json',
    'dissertation_artifacts/artifact_manifest_sha256.json',
]


def sha256_file(path: Path) -> str:
    if not path.exists() or not path.is_file():
        return ''
    h = hashlib.sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def run_command(cmd: list[str]) -> dict[str, Any]:
    proc = subprocess.run(cmd, cwd=ROOT, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, check=False)
    return {'command': ' '.join(cmd), 'returncode': proc.returncode, 'output_tail': proc.stdout[-4000:]}


def run() -> dict[str, Any]:
    runs = [run_command(cmd) for cmd in COMMANDS]
    failed = [r for r in runs if r['returncode'] != 0]
    if failed:
        raise SystemExit(json.dumps({'status': 'failed', 'failed': failed}, ensure_ascii=False, indent=2))
    subprocess.run(['make', 'chapter3-artifacts'], cwd=ROOT, check=False, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    subprocess.run([str(ROOT / 'Makefile')], cwd=ROOT, check=False, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True) if False else None
    files = [{'path': rel, 'sha256': sha256_file(ROOT / rel), 'exists': (ROOT / rel).exists()} for rel in KEY_FILES]
    manifest = {'status': 'ok', 'checked_at': datetime.now(timezone.utc).isoformat(), 'commands': runs, 'files': files}
    manifest_path = ROOT / 'evidence' / 'doctoral_final_manifest_sha256.json'
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding='utf-8')
    lines = ['# Doctoral final evidence index', '', f"checked_at: `{manifest['checked_at']}`", '', '## Commands', '']
    for r in runs:
        lines.append(f"- `{r['command']}` -> `{r['returncode']}`")
    lines += ['', '## Key files', '', '| file | exists | sha256 |', '| --- | ---: | --- |']
    for f in files:
        lines.append(f"| `{f['path']}` | `{f['exists']}` | `{f['sha256']}` |")
    index_path = ROOT / 'reports' / 'doctoral_final_evidence_index.md'
    index_path.write_text('\n'.join(lines) + '\n', encoding='utf-8')
    return {'status': 'ok', 'index': str(index_path), 'manifest': str(manifest_path), 'files': len(files)}


if __name__ == '__main__':
    print(json.dumps(run(), ensure_ascii=False, indent=2))
