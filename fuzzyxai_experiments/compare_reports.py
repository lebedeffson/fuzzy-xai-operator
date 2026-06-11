from __future__ import annotations

import csv
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent
REQUIRED_FILES = [
    'reports/chapter5/hybrid_xiris_summary.json',
    'reports/chapter5/hybrid_xiris_objects.csv',
    'reports/chapter5/hybrid_xiris_blocking_case.json',
    'reports/chapter5/beacon_xai_summary.json',
    'reports/chapter5/gis_integro_route_metrics.json',
    'reports/chapter5/gd_anfis_shap_report.json',
    'reports/chapter5/scenario_run_summary.csv',
    'reports/chapter5/scenario_baseline_comparison.csv',
    'tables/generated_tables.tex',
    'tables/scenario_summary.md',
    'tables/hybrid_xiris_baseline_comparison.md',
    'tables/beacon_xai_summary.md',
    'tables/gis_integro_metrics.md',
    'tables/gd_anfis_shap_metrics.md',
    'start_gui.sh',
    'export_gui_screenshots.sh',
    'gui_app.py',
    'reports/gui_screenshots/01_home_dashboard.png',
    'reports/gui_screenshots/07_gd_anfis_shap_route_report.png',
    'reports/gui_screenshots/08_evidence_center.png',
    'reports/gui_screenshots/09_developer_details.png',
]


def _numeric_only(obj: Any) -> Any:
    if isinstance(obj, dict):
        return {k: _numeric_only(v) for k, v in sorted(obj.items()) if k not in {'generated_at', 'runtime_seconds', 'processing_time_seconds'}}
    if isinstance(obj, list):
        return [_numeric_only(v) for v in obj]
    if isinstance(obj, float):
        return round(obj, 6)
    return obj


def report_hash(path: Path) -> str:
    """Hash stable numeric/report fields with timestamp/runtime ignored."""
    payload = _numeric_only(json.loads(path.read_text(encoding='utf-8')))
    return hashlib.sha256(json.dumps(payload, sort_keys=True, ensure_ascii=False).encode('utf-8')).hexdigest()


def _read(rel: str) -> dict[str, Any]:
    return json.loads((ROOT / rel).read_text(encoding='utf-8'))


def _csv_rows(rel: str) -> list[dict[str, str]]:
    with (ROOT / rel).open(encoding='utf-8') as f:
        return list(csv.DictReader(f))


def _require(name: str, condition: bool) -> None:
    if not condition:
        raise SystemExit(f'FAIL: {name}')
    print(f'PASS: {name}')


def validate_default() -> None:
    missing = [p for p in REQUIRED_FILES if not (ROOT / p).exists()]
    if missing:
        raise SystemExit(f"FAIL: required_files missing={missing}")

    hybrid = _read('reports/chapter5/hybrid_xiris_summary.json')
    hybrid_rows = _csv_rows('reports/chapter5/hybrid_xiris_objects.csv')
    block = _read('reports/chapter5/hybrid_xiris_blocking_case.json')
    _require(
        'hybrid_xiris',
        hybrid['total_objects'] == 1000
        and hybrid['critical_cases'] == 168
        and hybrid['baseline_missed'] == 168
        and hybrid['fuzzyxai_missed'] == 0
        and len(hybrid_rows) == 1000
        and {'chi_R_crit', 'chi_Auto', 'reason'} <= set(hybrid_rows[0])
        and block['chi_R_crit'] == 1
        and block['chi_Auto'] is False
        and block['action'] == 'block',
    )

    beacon = _read('reports/chapter5/beacon_xai_summary.json')
    _require('beacon_xai', beacon['total_signals'] == 100 and beacon['valid_after_adapter'] == 83 and beacon['baseline_manual_checks'] == 64 and beacon['fuzzyxai_manual_checks'] == 11 and beacon['audit_reports'] == 12)

    gis = _read('reports/chapter5/gis_integro_route_metrics.json')
    _require('gis_integro', gis['probability'] == 0.67 and gis['mean_alpha_k'] == 0.72 and gis['positive_SHAP_support'] == 0.47 and gis['gamma_route'] == 0.2 and gis['Delta'] == 0.08)

    gd = _read('reports/chapter5/gd_anfis_shap_report.json')
    _require('gd_anfis_shap', gd['n_rules'] == 3 and gd['Delta'] == 0.16 and gd['I_pre'] == 0.71 and gd['action'] == 'audit_report')

    manifest = _read('manifest_sha256.json')
    _require('generated_tables', (ROOT / 'tables/generated_tables.tex').exists())
    _require('checksums', len(manifest.get('files', [])) > 0 and (ROOT / 'checksums.sha256').exists())


def main() -> None:
    """Validate reports by default or print hashes for paths passed on CLI."""
    if len(sys.argv) == 1:
        validate_default()
        return
    for arg in sys.argv[1:]:
        path = Path(arg)
        print(f'{path}: {report_hash(path)}')


if __name__ == '__main__':
    main()
