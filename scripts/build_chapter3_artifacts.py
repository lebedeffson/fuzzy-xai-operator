from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding='utf-8')


def run(out_dir: str | Path = 'reports/chapter3') -> dict[str, Any]:
    out = ROOT / out_dir
    specs = ROOT / 'dissertation_artifacts/diagram_specs/chapter3'
    out.mkdir(parents=True, exist_ok=True); specs.mkdir(parents=True, exist_ok=True)
    rows = [
        {'dataset': 'breast_cancer', 'role': 'quantitative_demo', 'used_for': 'operator_metrics', 'critical_ruptures_present': False, 'notes': 'real medical tabular dataset'},
        {'dataset': 'wine_risk', 'role': 'transfer_demo', 'used_for': 'tabular_portability', 'critical_ruptures_present': False, 'notes': 'non-medical tabular transfer'},
        {'dataset': 'diabetes_binary', 'role': 'calibration_stress_test', 'used_for': 'borderline_uncertainty', 'critical_ruptures_present': False, 'notes': 'stress-test for domain thresholds'},
        {'dataset': 'synthetic_ruptures', 'role': 'diagnostic_benchmark', 'used_for': 'safety_validation', 'critical_ruptures_present': True, 'notes': 'controlled rupture injection'},
        {'dataset': 'registry_fixtures', 'role': 'transfer_demo', 'used_for': 'ecosystem_portability', 'critical_ruptures_present': False, 'notes': 'adapter-level evidence'},
    ]
    with (out / 'dataset_roles_summary.csv').open('w', encoding='utf-8', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader(); writer.writerows(rows)
    md = ['# Dataset roles summary', '', '| dataset | role | used_for | critical ruptures | notes |', '| --- | --- | --- | ---: | --- |']
    for r in rows:
        md.append(f"| `{r['dataset']}` | `{r['role']}` | `{r['used_for']}` | `{r['critical_ruptures_present']}` | {r['notes']} |")
    (out / 'dataset_roles_summary.md').write_text('\n'.join(md) + '\n', encoding='utf-8')
    (out / 'safety_limitation_insert.md').write_text(
        'Safety-результаты по критическим разрывам проверяются на диагностическом стенде `synthetic_ruptures`, где `chi_R^crit` вносится контролируемо и должен приводить к `block`. '
        'На реальных медицинских табличных данных критические разрывы такого типа не размечены экспертно; поэтому Breast Cancer/Wine/Diabetes используются для проверки контура, калибровки и переносимости, а не для клинического доказательства безопасности. '
        'Полная предметная валидация требует отдельной экспертной разметки и регулируемого протокола.\n',
        encoding='utf-8',
    )
    diagrams = {
        'fig_3_2_hierarchy.json': {
            'figure_id': 'fig_3_2', 'title': 'Hierarchy of fuzzy uncertainty representations',
            'nodes': [{'id': 'F0', 'label': 'F0'}, {'id': 'F_int', 'label': 'F_int'}, {'id': 'F_H', 'label': 'F_H'}, {'id': 'F_N_src', 'label': 'F_N^src'}, {'id': 'F_ML', 'label': 'F_ML-audit'}],
            'edges': [{'from': 'F0', 'to': 'F_int'}, {'from': 'F0', 'to': 'F_H'}, {'from': 'F_int', 'to': 'F_N_src'}, {'from': 'F_H', 'to': 'F_ML'}, {'from': 'F_N_src', 'to': 'F_ML'}],
            'color_roles': {'selected': '#0f766e', 'candidate': '#dbeafe'}, 'caption': 'Иерархия классов представления неопределенности.', 'short_explanation': 'Higher classes preserve more structural uncertainty.'
        },
        'fig_3_3_reduction.json': {
            'figure_id': 'fig_3_3', 'title': 'Reduction to F0 and Delta',
            'nodes': [{'id': 'A_F', 'label': 'A_k^F'}, {'id': 'reduce', 'label': 'reduce'}, {'id': 'F0', 'label': 'F0'}, {'id': 'Delta', 'label': 'Delta'}],
            'edges': [{'from': 'A_F', 'to': 'reduce'}, {'from': 'reduce', 'to': 'F0'}, {'from': 'reduce', 'to': 'Delta'}],
            'color_roles': {'loss': '#dc2626', 'representation': '#2563eb'}, 'caption': 'Редукция расширенного представления и измерение потери Delta.', 'short_explanation': 'Delta controls how much structure is lost.'
        },
        'fig_3_8_chi_auto_sample113.json': {
            'figure_id': 'fig_3_8', 'title': 'Chi_Auto for sample_113',
            'nodes': [{'id': 'e_model', 'label': 'E_model^ext', 'type': 'explanation'}, {'id': 'e_risk', 'label': 'E_risk', 'type': 'risk'}, {'id': 'e_action', 'label': 'E_action', 'type': 'action'}],
            'edges': [{'from': 'e_model', 'to': 'e_risk', 'label': 'f'}, {'from': 'e_risk', 'to': 'e_action', 'label': 'g'}],
            'color_roles': {'blocked': '#dc2626', 'allowed': '#16a34a'}, 'caption': 'Контекстная проверка chi_Auto для sample_113.', 'short_explanation': 'chi_auto=false because auto maps to audit, which is absent in AutoAccept.'
        },
        'fig_3_4_route.json': {
            'figure_id': 'fig_3_4', 'title': 'Risk-aware observer route',
            'nodes': [{'id': x, 'label': x} for x in ['DatasetCase', 'Prediction', 'E_k', 'A_k^F', 'CertifiedPath/Rupture', 'chi_Auto', 'rho', 'Action']],
            'edges': [{'from': a, 'to': b} for a, b in zip(['DatasetCase', 'Prediction', 'E_k', 'A_k^F', 'CertifiedPath/Rupture', 'chi_Auto', 'rho'], ['Prediction', 'E_k', 'A_k^F', 'CertifiedPath/Rupture', 'chi_Auto', 'rho', 'Action'])],
            'color_roles': {'risk': '#ea580c', 'action': '#0f766e'}, 'caption': 'Единый маршрут наблюдателя от данных к действию.', 'short_explanation': 'Shows where rupture, context and risk enter the route.'
        },
    }
    for name, payload in diagrams.items():
        _write_json(specs / name, payload)
    return {'status': 'ok', 'rows': len(rows), 'diagram_specs': len(diagrams)}


def main() -> None:
    parser = argparse.ArgumentParser(); parser.add_argument('--out-dir', default='reports/chapter3')
    args = parser.parse_args(); print(json.dumps(run(args.out_dir), ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
