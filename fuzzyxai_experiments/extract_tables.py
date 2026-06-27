from __future__ import annotations

import json
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent
TABLES = ROOT / 'tables'


def read_json(rel: str) -> dict[str, Any]:
    return json.loads((ROOT / rel).read_text(encoding='utf-8'))


def md_table(rows: list[dict[str, Any]], cols: list[str]) -> str:
    out = ['| ' + ' | '.join(cols) + ' |', '| ' + ' | '.join(['---'] * len(cols)) + ' |']
    for row in rows:
        out.append('| ' + ' | '.join(str(row.get(c, '')) for c in cols) + ' |')
    return '\n'.join(out) + '\n'


def write(name: str, rows: list[dict[str, Any]], cols: list[str]) -> None:
    TABLES.mkdir(parents=True, exist_ok=True)
    (TABLES / name).write_text(md_table(rows, cols), encoding='utf-8')


def main() -> None:
    hybrid = read_json('reports/chapter5/hybrid_xiris_summary.json')
    beacon = read_json('reports/chapter5/beacon_xai_summary.json')
    gis = read_json('reports/chapter5/gis_integro_route_metrics.json')
    gd = read_json('reports/chapter5/gd_anfis_shap_report.json')
    registry = read_json('registry/modules.json')['modules']

    write('scenario_summary.md', registry, ['registry_id', 'status', 'adapter', 'input_artifact', 'output_report', 'claim_scope'])
    write('hybrid_xiris_baseline_comparison.md', [hybrid], ['total_objects', 'critical_cases', 'baseline_missed', 'fuzzyxai_missed', 'false_block'])
    write('beacon_xai_summary.md', [beacon], ['total_signals', 'valid_after_adapter', 'baseline_manual_checks', 'fuzzyxai_manual_checks', 'audit_reports'])
    write('gis_integro_metrics.md', [gis], ['probability', 'mean_alpha_k', 'positive_SHAP_support', 'gamma_route', 'Delta', 'source_status'])
    write('gd_anfis_shap_metrics.md', [gd], ['registry_id', 'n_rules', 'Delta', 'I_pre', 'action', 'source_status'])


if __name__ == '__main__':
    main()
