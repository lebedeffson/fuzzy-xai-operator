from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent


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


def _require(name: str, condition: bool) -> None:
    if not condition:
        raise SystemExit(f'FAIL: {name}')
    print(f'PASS: {name}')


def validate_default() -> None:
    hybrid = _read('reports/chapter5/hybrid_xiris_summary.json')
    _require('hybrid_xiris', hybrid['total_objects'] == 1000 and hybrid['critical_cases'] == 168 and hybrid['baseline_missed'] == 168 and hybrid['fuzzyxai_missed'] == 0)

    beacon = _read('reports/chapter5/beacon_xai_summary.json')
    _require('beacon_xai', beacon['total_signals'] == 100 and beacon['valid_after_adapter'] == 83 and beacon['baseline_manual_checks'] == 64 and beacon['fuzzyxai_manual_checks'] == 11 and beacon['audit_reports'] == 12)

    gis = _read('reports/chapter5/gis_integro_route_metrics.json')
    _require('gis_integro', gis['probability'] == 0.67 and gis['mean_alpha_k'] == 0.72 and gis['positive_SHAP_support'] == 0.47 and gis['gamma_route'] == 0.2 and gis['Delta'] == 0.08)

    gd = _read('reports/chapter5/gd_anfis_shap_report.json')
    _require('gd_anfis_shap', gd['n_rules'] == 3 and gd['action'] == 'audit_report' and gd['I_pre'] > 0)

    _require('generated_tables', (ROOT / 'tables/generated_tables.tex').exists())
    manifest = _read('manifest_sha256.json')
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
