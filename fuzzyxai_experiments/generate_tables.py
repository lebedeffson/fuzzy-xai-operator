from __future__ import annotations

import json
from pathlib import Path
from typing import Any

REPORTS = Path(__file__).resolve().parent / 'reports'


def _load(name: str) -> dict[str, Any]:
    return json.loads((REPORTS / name).read_text(encoding='utf-8'))


def _latex(rows: list[dict[str, Any]], cols: list[str]) -> str:
    lines = ['\\begin{tabular}{' + 'l' * len(cols) + '}', ' & '.join(cols) + ' \\', '\\hline']
    for row in rows:
        lines.append(' & '.join(str(row.get(c, '')) for c in cols) + ' \\')
    lines.append('\\end{tabular}')
    return '\n'.join(lines)


def main() -> None:
    """Print thesis-ready LaTeX snippets from generated JSON reports."""
    critical = _load('ch3_diagnostic_stand.json')
    print('% Table: controlled critical ruptures')
    print(_latex(critical['rows'], ['policy', 'n_objects', 'missed_critical_ruptures', 'critical_rupture_recall', 'agreement_reference']))

    hybrid = _load('ch5_hybrid.json')
    print('\n% HYBRID-XIRIS baseline vs FuzzyXAI')
    print(_latex([hybrid], ['total_objects', 'critical_cases', 'baseline_missed', 'fuzzyxai_missed', 'false_block']))

    beacon = _load('ch5_beacon.json')
    print('\n% BEACON-XAI summary')
    print(_latex([beacon], ['total_signals', 'valid_after_adapter', 'baseline_manual_checks', 'fuzzyxai_manual_checks', 'audit_reports']))

    gis = _load('ch5_gis.json')['metrics']
    print('\n% GIS INTEGRO route metrics')
    print(_latex([gis], ['registry_id', 'gamma_route', 'Delta', 'probability', 'mean_alpha_k', 'positive_SHAP_support']))

    gd = _load('ch5_gd_anfis_shap.json')
    print('\n% GD-ANFIS/SHAP route metrics')
    print(_latex([gd], ['registry_id', 'n_rules', 'Delta', 'I_pre', 'action', 'source_status']))

    print('\n% Evidence file map')
    rows = [
        {'scenario': 'HYBRID-XIRIS', 'evidence_file': 'reports/chapter5/hybrid_xiris_summary.json'},
        {'scenario': 'BEACON-XAI', 'evidence_file': 'reports/chapter5/beacon_xai_summary.json'},
        {'scenario': 'GIS INTEGRO', 'evidence_file': 'reports/chapter5/gis_integro_route_metrics.json'},
        {'scenario': 'GD-ANFIS/SHAP', 'evidence_file': 'reports/chapter5/gd_anfis_shap_report.json'},
    ]
    print(_latex(rows, ['scenario', 'evidence_file']))


if __name__ == '__main__':
    main()
