from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from apps.services.layered_case import LayeredCaseService, build_case_state
from fuzzyxai.datasets import list_dataset_modes, load_dataset_mode


ROOT = Path(__file__).resolve().parents[1]


def _fmt(v: Any) -> str:
    if v is None:
        return 'N/A'
    if isinstance(v, float):
        if v != v:
            return 'NaN'
        return f'{v:.6f}'.rstrip('0').rstrip('.')
    return str(v)


def _tex_escape(text: str) -> str:
    return (
        text.replace('\\', '\\textbackslash{}')
        .replace('_', '\\_')
        .replace('%', '\\%')
        .replace('&', '\\&')
        .replace('#', '\\#')
    )


def _load_dataset_rows() -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for spec in list_dataset_modes():
        status = 'READY'
        rows_n: int | None = None
        note = ''
        try:
            _record, df = load_dataset_mode(spec.key)
            rows_n = int(len(df))
        except FileNotFoundError as exc:
            status = 'MISSING'
            note = str(exc)
        except Exception as exc:
            status = 'ERROR'
            note = f'{type(exc).__name__}: {exc}'
        summary_path = ROOT / f'reports/datasets/{spec.key}/summary.json'
        summary = None
        if summary_path.exists():
            summary = json.loads(summary_path.read_text(encoding='utf-8'))
        rows.append(
            {
                'dataset_mode': spec.key,
                'domain': spec.domain,
                'validates': spec.purpose,
                'status': status,
                'rows': rows_n,
                'summary': summary,
                'note': note,
            }
        )
    return rows


def _component_rows() -> list[dict[str, str]]:
    return [
        {
            'stage': 'Dataset',
            'object': 'DatasetCase',
            'input': 'raw/local dataset mode',
            'output': 'normalized case data',
            'metric': 'status, rows, domain',
            'code': 'fuzzyxai/datasets/*, apps/services/layered_case.py',
            'explain': 'Подготовка данных в единый формат для последующих слоёв.',
        },
        {
            'stage': 'Model',
            'object': 'prediction, p(y=1), uncertainty',
            'input': 'features(x)',
            'output': 'prediction, predicted_risk',
            'metric': 'accuracy, roc_auc',
            'code': 'fuzzyxai/pipelines/dataset_observer_pipeline.py',
            'explain': 'Базовый прогноз без принятия финального решения.',
        },
        {
            'stage': 'Omega',
            'object': 'E_model=<L,mu,R,alpha,u,tau>',
            'input': 'model output',
            'output': 'structured explanation object',
            'metric': 'u, active rules',
            'code': 'fuzzyxai/core/system_operator.py',
            'explain': 'Перевод численного прогноза в проверяемую объяснительную структуру.',
        },
        {
            'stage': 'Expl / Rupture',
            'object': 'T_ij or D_ij',
            'input': 'E_model -> E_risk -> E_action',
            'output': 'morphism / rupture',
            'metric': 'gamma, gamma_max, chi_R',
            'code': 'fuzzyxai/core/composition.py, fuzzyxai/core/diagnostics.py',
            'explain': 'Проверка согласованности переходов между слоями.',
        },
        {
            'stage': 'Fuzzy Representation',
            'object': 'F0 / F_int / F_H / F_N_src / FML-audit',
            'input': 'uncertainty profile',
            'output': 'selected_class, Delta',
            'metric': 'coverage, complexity, reduction_loss',
            'code': 'fuzzyxai/risk/representation_selection.py, apps/services/layered_case.py',
            'explain': 'Выбор минимально достаточного класса неопределенности.',
        },
        {
            'stage': 'Context Topos',
            'object': 'RiskContext, AutoAccept, chi_Auto',
            'input': 'context presheaves',
            'output': 'auto apply allowed/denied',
            'metric': 'AutoAccept(E)',
            'code': 'fuzzyxai/category/context_topos.py, fuzzyxai/category/subpresheaf.py',
            'explain': 'Контекстно-зависимый фильтр для безопасного автоприменения.',
        },
        {
            'stage': 'Risk Observer',
            'object': 'rho, chi_R, chi_R^crit, action',
            'input': 'risk components',
            'output': 'accept/lower/request/defer/block',
            'metric': 'mean_rho, action_distribution',
            'code': 'fuzzyxai/risk/*, apps/layered_demo.py',
            'explain': 'Интегральный риск и пороговая политика безопасного действия.',
        },
        {
            'stage': 'Reports',
            'object': 'JSON/CSV/MD',
            'input': 'pipeline artifacts',
            'output': 'dissertation tables',
            'metric': 'reproducibility',
            'code': 'experiments/dataset_benchmark.py, experiments/dissertation_demo_summary.py',
            'explain': 'Фиксация воспроизводимых результатов для защиты.',
        },
    ]


def _case_rows() -> list[dict[str, Any]]:
    service = LayeredCaseService.create()
    cases = [('safe', 'breast_cancer'), ('ambiguous', 'breast_cancer'), ('rupture', 'breast_cancer')]
    rows: list[dict[str, Any]] = []
    for scenario, dataset_mode in cases:
        st = build_case_state(service, scenario, dataset_mode=dataset_mode)
        rows.append(
            {
                'scenario': scenario,
                'dataset': dataset_mode,
                'prediction': st['model']['prediction'],
                'predicted_risk': round(float(st['model']['predicted_risk']), 6),
                'uncertainty': round(float(st['model']['uncertainty']), 6),
                'selected_class': st['uncertainty']['selected_class'],
                'delta': round(float(st['uncertainty']['delta']), 6),
                'i_pre': round(float(st['explanation']['I_pre']), 6),
                'rho': round(float(st['risk']['rho']), 6),
                'chi_R': int(st['risk'].get('chi_R', 0)),
                'chi_R_crit': int(st['risk'].get('chi_R_crit', 0)),
                'action': st['risk']['action'],
            }
        )
    return rows


def _md_table(headers: list[str], rows: list[list[Any]]) -> str:
    out = ['| ' + ' | '.join(headers) + ' |', '| ' + ' | '.join(['---'] * len(headers)) + ' |']
    for row in rows:
        out.append('| ' + ' | '.join(str(x) for x in row) + ' |')
    return '\n'.join(out)


def _latex_table(headers: list[str], rows: list[list[Any]], caption: str, label: str) -> str:
    col_spec = '|'.join(['l'] * len(headers))
    out = [
        '\\begin{table}[h]',
        '\\centering',
        f'\\caption{{{_tex_escape(caption)}}}',
        f'\\label{{{_tex_escape(label)}}}',
        f'\\begin{{tabular}}{{|{col_spec}|}}',
        '\\hline',
        ' & '.join(_tex_escape(h) for h in headers) + ' \\\\',
        '\\hline',
    ]
    for row in rows:
        out.append(' & '.join(_tex_escape(str(x)) for x in row) + ' \\\\')
        out.append('\\hline')
    out += ['\\end{tabular}', '\\end{table}', '']
    return '\n'.join(out)


def generate(out_dir: Path) -> dict[str, Any]:
    out_dir.mkdir(parents=True, exist_ok=True)
    dataset_rows = _load_dataset_rows()
    component_rows = _component_rows()
    case_rows = _case_rows()
    quantitative_keys = {'breast_cancer', 'diabetes_binary', 'wine_risk', 'synthetic_ruptures'}
    quantitative_rows = [r for r in dataset_rows if r['dataset_mode'] in quantitative_keys]
    registry_rows = [r for r in dataset_rows if r['dataset_mode'].startswith('registry_')]

    payload = {
        'components': component_rows,
        'datasets': dataset_rows,
        'quantitative_datasets': quantitative_rows,
        'registry_datasets': registry_rows,
        'observer_cases': case_rows,
        'policy': [
            {'condition': 'chi_R^crit(x)=1', 'action': 'block', 'meaning': 'критический разрыв'},
            {'condition': 'rho(x)>=0.80 and chi_R^crit(x)=0', 'action': 'defer_to_human', 'meaning': 'высокий риск без крит. разрыва'},
            {'condition': 'rho(x)<0.80 and chi_R(x)=1 and chi_R^crit(x)=0', 'action': 'request_more_data', 'meaning': 'некритический разрыв'},
        ],
    }

    (out_dir / 'dissertation_component_tables.json').write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding='utf-8'
    )

    md_lines: list[str] = ['# Полные таблицы по компонентам системы FuzzyXAI', '']
    md_lines += ['## 1) Компоненты системы', '']
    md_lines.append(
        _md_table(
            ['Этап', 'Объект', 'Вход', 'Выход', 'Метрика', 'Код', 'Пояснение'],
            [
                [r['stage'], r['object'], r['input'], r['output'], r['metric'], r['code'], r['explain']]
                for r in component_rows
            ],
        )
    )
    md_lines += ['', '## 2) Датасеты и готовность', '']
    md_lines.append(
        _md_table(
            ['Dataset mode', 'Domain', 'Rows', 'Status', 'Pipeline', 'What validates'],
            [
                [
                    r['dataset_mode'],
                    r['domain'],
                    r['rows'] if r['rows'] is not None else '-',
                    r['status'],
                    _fmt((r['summary'] or {}).get('pipeline_completed', None)),
                    r['validates'],
                ]
                for r in dataset_rows
            ],
        )
    )
    md_lines += ['', '## 3) Количественная проверка (встроенные режимы)', '']
    md_lines.append(
        _md_table(
            [
                'Dataset',
                'Acc',
                'ROC AUC',
                'Agreement proxy',
                'Rupture rate',
                'Crit rupture rate',
                'positive_rate',
                'score_std',
                'Use tag',
                'Note',
            ],
            [
                [
                    r['dataset_mode'],
                    _fmt((r['summary'] or {}).get('model_accuracy', None)),
                    _fmt((r['summary'] or {}).get('model_roc_auc', None)),
                    _fmt((r['summary'] or {}).get('agreement_proxy', None)),
                    _fmt((r['summary'] or {}).get('rupture_rate', None)),
                    _fmt((r['summary'] or {}).get('critical_rupture_rate', None)),
                    _fmt((r['summary'] or {}).get('positive_rate', None)),
                    _fmt((r['summary'] or {}).get('score_std', None)),
                    _fmt((r['summary'] or {}).get('recommended_use_in_dissertation', None)),
                    str((r['summary'] or {}).get('notes', '-')),
                ]
                for r in quantitative_rows
            ],
        )
    )
    md_lines += ['', '## 4) Registry-режимы: readiness и ограничения интерпретации', '']
    md_lines.append(
        _md_table(
            ['Dataset', 'Pipeline', 'Agreement proxy applicable', 'Agreement proxy', 'ROC reason', 'Limitation'],
            [
                [
                    r['dataset_mode'],
                    _fmt((r['summary'] or {}).get('pipeline_completed', None)),
                    _fmt((r['summary'] or {}).get('agreement_proxy_applicable', None)),
                    _fmt((r['summary'] or {}).get('agreement_proxy', None)),
                    _fmt((r['summary'] or {}).get('reason_if_roc_auc_nan_or_05', None)),
                    _fmt((r['summary'] or {}).get('notes', None)),
                ]
                for r in registry_rows
            ],
        )
    )
    md_lines += ['', '## 5) Кейсы наблюдателя по слоям', '']
    md_lines.append(
        _md_table(
            ['Scenario', 'predicted_risk', 'uncertainty', 'selected_class', 'Delta', 'I_pre', 'rho', 'chi_R', 'chi_R_crit', 'action'],
            [
                [r['scenario'], r['predicted_risk'], r['uncertainty'], r['selected_class'], r['delta'], r['i_pre'], r['rho'], r['chi_R'], r['chi_R_crit'], r['action']]
                for r in case_rows
            ],
        )
    )
    md_lines += ['', '## 6) Пороговая политика', '']
    md_lines.append(
        _md_table(
            ['Условие', 'Действие', 'Пояснение'],
            [[r['condition'], r['action'], r['meaning']] for r in payload['policy']],
        )
    )
    md_lines.append('')
    md_lines.append('Примечание: для MosMed оригинальный архив ~104.9GB, в локальном контуре используется малый табличный аудит-слепок.')

    (out_dir / 'dissertation_component_tables.md').write_text('\n'.join(md_lines), encoding='utf-8')

    readiness_tex_headers = ['Mode', 'Domain', 'Rows', 'Status', 'Pipeline', 'Validates']
    readiness_tex_rows = [
        [
            r['dataset_mode'],
            r['domain'],
            r['rows'] if r['rows'] is not None else '-',
            r['status'],
            _fmt((r['summary'] or {}).get('pipeline_completed', None)),
            r['validates'],
        ]
        for r in dataset_rows
    ]
    quantitative_tex_headers = ['Dataset', 'Acc', 'ROC AUC', 'Agreement proxy', 'Rupture', 'Crit rupture', 'positive_rate', 'note']
    quantitative_tex_rows = [
        [
            r['dataset_mode'],
            _fmt((r['summary'] or {}).get('model_accuracy', None)),
            _fmt((r['summary'] or {}).get('model_roc_auc', None)),
            _fmt((r['summary'] or {}).get('agreement_proxy', None)),
            _fmt((r['summary'] or {}).get('rupture_rate', None)),
            _fmt((r['summary'] or {}).get('critical_rupture_rate', None)),
            _fmt((r['summary'] or {}).get('positive_rate', None)),
            _fmt((r['summary'] or {}).get('reason_if_roc_auc_nan_or_05', None)),
        ]
        for r in quantitative_rows
    ]
    (out_dir / 'dissertation_table_readiness.tex').write_text(
        _latex_table(readiness_tex_headers, readiness_tex_rows, 'Dataset readiness modes', 'tab:dataset_readiness'),
        encoding='utf-8',
    )
    (out_dir / 'dissertation_table_quantitative.tex').write_text(
        _latex_table(
            quantitative_tex_headers,
            quantitative_tex_rows,
            'Quantitative validation on built-in datasets',
            'tab:dataset_quantitative',
        ),
        encoding='utf-8',
    )
    return payload


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--out-dir', default='reports')
    args = parser.parse_args()
    payload = generate(Path(args.out_dir))
    print(json.dumps({'status': 'ok', 'out_dir': str(Path(args.out_dir).resolve()), 'components': len(payload['components'])}, ensure_ascii=False))


if __name__ == '__main__':
    main()
