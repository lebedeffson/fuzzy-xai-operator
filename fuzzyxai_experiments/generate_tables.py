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
    gis = _load('ch5_gis.json')['metrics']
    print('\n% GIS INTEGRO route metrics')
    print(_latex([gis], ['registry_id', 'gamma_route', 'Delta', 'probability', 'mean_alpha_k', 'positive_shap_support']))


if __name__ == '__main__':
    main()
