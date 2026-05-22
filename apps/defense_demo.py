"""Focused defense demo for dissertation chapters 2 and 3.

Run from repository root:
    python apps/defense_demo.py --port 8085
"""
from __future__ import annotations

import argparse
import io
import json
import sys
import traceback
from pathlib import Path
from typing import Any, Dict, Mapping

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

try:
    import pandas as pd
    import plotly.graph_objects as go
    from nicegui import ui
except Exception as exc:  # pragma: no cover
    raise SystemExit('Install requirements.txt first: pandas, plotly, nicegui are required.') from exc

from fuzzyxai.core.plan_builder import build_explain_plan_from_dataframe
from fuzzyxai.demo.synthetic import (
    build_demo_composition,
    build_demo_explanation,
    default_candidates,
    metadata_for_demo,
    sample_dataframe,
)
from fuzzyxai.selection.pareto_selector import pareto_front, select_minimal_sufficient
from fuzzyxai.visual.composition_graph import edge_report
from fuzzyxai.visual.representation_plots import explainplan_membership_figure, representation_figure

REPORTS = ROOT / 'reports'
REPORTS.mkdir(exist_ok=True)


STATE: Dict[str, Any] = {
    'df': sample_dataframe(),
    'risk': 0.72,
    'feature': 'risk_score',
    'conflict': False,
    'mode': 'audit',
    'plan': None,
    'explanation': None,
    'composition': None,
}


def build_plan_from_state() -> None:
    df = STATE['df']
    target = 'target' if 'target' in df.columns else df.columns[-1]
    STATE['plan'] = build_explain_plan_from_dataframe(df, target=target, mode=STATE['mode']).with_reduction_weight(0.10)
    numeric = STATE['plan'].metadata.get('numeric_features', [])
    if STATE.get('feature') not in numeric and numeric:
        STATE['feature'] = numeric[0]


def recompute() -> None:
    if STATE.get('plan') is None:
        build_plan_from_state()
    metadata = metadata_for_demo(intervals=True, experts=2, conflict=True, audit=STATE['mode'] == 'audit')
    STATE['explanation'] = build_demo_explanation(
        float(STATE['risk']),
        plan=STATE['plan'],
        metadata=metadata,
        audience='doctor',
        use_llm=False,
    )
    STATE['composition'] = build_demo_composition(
        float(STATE['risk']),
        min(1.0, float(STATE['risk']) + 0.02),
        plan=STATE['plan'],
        conflict=bool(STATE['conflict']),
    )


def safe(fn, *, where: str) -> None:
    try:
        fn()
    except Exception as exc:  # pragma: no cover - visible GUI safety net
        ui.notify(f'{where}: {exc}', type='negative', timeout=7000)
        (REPORTS / 'defense_demo_last_error.txt').write_text(traceback.format_exc(), encoding='utf-8')


def apply_style() -> None:
    ui.add_head_html(
        """
        <style>
        :root {
          --bg: #f6f7f9;
          --panel: #ffffff;
          --text: #16202a;
          --muted: #637083;
          --line: #d9dee7;
          --blue: #2563eb;
          --green: #0f9f6e;
          --amber: #c47b00;
          --red: #d83a3a;
        }
        body { background: var(--bg); color: var(--text); }
        .fx-shell { max-width: 1440px; margin: 0 auto; }
        .fx-topbar {
          background: rgba(255,255,255,0.96); border-bottom: 1px solid var(--line);
          position: sticky; top: 0; z-index: 20;
          backdrop-filter: blur(10px);
        }
        .fx-panel {
          background: var(--panel); border: 1px solid var(--line);
          border-radius: 8px; padding: 18px;
          box-shadow: 0 10px 26px rgba(22, 32, 42, 0.06);
        }
        .fx-metric {
          background: #f9fafb; border: 1px solid #e5e7eb;
          border-radius: 8px; padding: 12px; min-width: 150px; flex: 1;
        }
        .fx-muted { color: var(--muted); }
        .fx-title { font-weight: 760; letter-spacing: 0; }
        .fx-chip {
          display: inline-block; border-radius: 999px; padding: 3px 8px;
          background: #eef2ff; color: #1f3a8a; font-size: 12px; font-weight: 650;
        }
        .fx-chip-green { background: #e7f8f1; color: #08684a; }
        .fx-chip-red { background: #fdecec; color: #9f2323; }
        .fx-note {
          background: #f8fafc; border: 1px solid #e2e8f0;
          border-radius: 8px; padding: 10px; color: #334155;
        }
        .fx-status-ok {
          background: #e7f8f1; border: 1px solid #a6e3c8;
          border-radius: 8px; padding: 12px; color: #08684a;
        }
        .fx-status-bad {
          background: #fdecec; border: 1px solid #f4b4b4;
          border-radius: 8px; padding: 12px; color: #9f2323;
        }
        .fx-step {
          border-left: 4px solid var(--blue); padding-left: 10px;
        }
        .q-table th, .q-table td { font-size: 12px; }
        .js-plotly-plot .plotly .main-svg { border-radius: 8px; }
        </style>
        """
    )


def metric(title: str, value: Any, caption: str = '') -> None:
    with ui.element('div').classes('fx-metric'):
        ui.label(title).classes('text-xs fx-muted')
        ui.label(str(value)).classes('text-xl fx-title')
        if caption:
            ui.label(caption).classes('text-xs fx-muted')


def row_metrics(items) -> None:
    with ui.row().classes('w-full gap-2'):
        for item in items:
            metric(*item)


def feature_options() -> list[str]:
    plan = STATE['plan']
    if plan is None:
        return []
    return list(plan.metadata.get('numeric_features', []))


def quantile_rows() -> list[dict[str, Any]]:
    plan = STATE['plan']
    feature = STATE['feature']
    terms = ((plan.metadata or {}).get('feature_terms', {}).get(feature, {}) if plan else {})
    q = terms.get('quantiles', {})
    term_specs = terms.get('terms', {})
    rows = []
    for name, spec in term_specs.items():
        rows.append({
            'term': name,
            'shape': spec.get('kind', ''),
            'left': round(float(spec.get('a', q.get('min', 0.0))), 4),
            'center': round(float(spec.get('b', q.get('median', 0.0))), 4),
            'right': round(float(spec.get('c', q.get('max', 0.0))), 4) if 'c' in spec else '',
        })
    return rows


def membership_bar() -> go.Figure:
    expl = STATE['explanation']
    memberships = expl['object'].metadata.get('memberships', {}) if expl else {}
    colors = {'low': '#0f9f6e', 'medium': '#c47b00', 'high': '#d83a3a'}
    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=list(memberships.keys()),
        y=list(memberships.values()),
        marker_color=[colors.get(k, '#2563eb') for k in memberships.keys()],
        text=[f'{v:.2f}' for v in memberships.values()],
        textposition='outside',
    ))
    fig.update_layout(
        title='Как этот риск попал в термы',
        xaxis_title='терм',
        yaxis_title='степень принадлежности',
        yaxis=dict(range=[0, 1.12], showgrid=True, gridcolor='#edf1f6'),
        xaxis=dict(showgrid=False),
        height=300,
        margin=dict(l=45, r=18, t=56, b=42),
        showlegend=False,
        plot_bgcolor='#ffffff',
        paper_bgcolor='#ffffff',
        font=dict(family='Arial, sans-serif', color='#16202a'),
    )
    return fig


def readable_class_name(name: str) -> str:
    labels = {
        'F0': 'простая нечёткость',
        'FI': 'интервалы',
        'FH': 'несколько экспертных оценок',
        'FNsrc': 'конфликт источников',
        'FML-audit': 'многоуровневое представление для аудита',
        'FML-user': 'многоуровневое представление для пользователя',
    }
    return labels.get(name, name)


def agreement_level(gamma: float) -> tuple[str, str, str]:
    if gamma < 0.20:
        return 'низкое', '#0f9f6e', 'компоненты почти не спорят'
    if gamma < 0.45:
        return 'среднее', '#c47b00', 'есть заметное расхождение'
    return 'высокое', '#d83a3a', 'нужна проверка'


def composition_story_figure(comp: Mapping[str, Any], rows: list[dict[str, Any]]) -> go.Figure:
    """Human-readable diagram for model-to-decision agreement."""
    comp_obj = comp['comp']
    diagnostic = getattr(comp_obj, 'code', None)
    gamma = float(rows[0]['gamma']) if rows else 0.0
    level, color, caption = agreement_level(gamma)
    left_terms = ', '.join(rows[0].get('left_terms', [])) if rows else 'n/a'
    right_terms = ', '.join(rows[0].get('right_terms', [])) if rows else 'n/a'
    status = 'РАЗРЫВ ОБНАРУЖЕН' if diagnostic else 'СОГЛАСОВАНО'
    status_color = '#d83a3a' if diagnostic else color

    fig = go.Figure()
    fig.add_shape(type='rect', x0=0.04, y0=0.30, x1=0.35, y1=0.82, line=dict(color='#cbd5e1', width=1.5), fillcolor='#ffffff')
    fig.add_shape(type='rect', x0=0.65, y0=0.30, x1=0.96, y1=0.82, line=dict(color='#cbd5e1', width=1.5), fillcolor='#ffffff')
    fig.add_shape(type='rect', x0=0.40, y0=0.43, x1=0.60, y1=0.61, line=dict(color=status_color, width=1), fillcolor='rgba(255,255,255,0.88)')
    fig.add_annotation(x=0.63, y=0.52, ax=0.37, ay=0.52, xref='x', yref='y', axref='x', ayref='y', showarrow=True, arrowhead=3, arrowsize=1.3, arrowwidth=5, arrowcolor=status_color)
    fig.add_annotation(x=0.50, y=0.56, text=f'{gamma:.3f}', showarrow=False, font=dict(color=status_color, size=20))
    fig.add_annotation(x=0.50, y=0.47, text='расхождение', showarrow=False, font=dict(color='#637083', size=11))
    fig.add_annotation(x=0.195, y=0.69, text='<b>Модель риска</b>', showarrow=False, font=dict(size=16, color='#16202a'))
    fig.add_annotation(x=0.195, y=0.53, text=f'термы<br>{left_terms}', showarrow=False, font=dict(size=12, color='#637083'))
    fig.add_annotation(x=0.805, y=0.69, text='<b>Модуль решения</b>', showarrow=False, font=dict(size=16, color='#16202a'))
    fig.add_annotation(x=0.805, y=0.53, text=f'термы<br>{right_terms}', showarrow=False, font=dict(size=12, color='#637083'))
    fig.add_annotation(x=0.50, y=0.30, text=caption, showarrow=False, font=dict(color='#637083', size=13))
    fig.add_annotation(
        x=0.50,
        y=0.13,
        text=f'<b>{status}</b>' if not diagnostic else f'<b>{status}</b>: {diagnostic}',
        showarrow=False,
        font=dict(color=status_color, size=20),
    )
    fig.update_layout(
        title='Проверка цепочки объяснения',
        xaxis=dict(visible=False, range=[0, 1]),
        yaxis=dict(visible=False, range=[0, 1]),
        height=340,
        margin=dict(l=16, r=16, t=54, b=16),
        plot_bgcolor='#ffffff',
        paper_bgcolor='#ffffff',
        font=dict(family='Arial, sans-serif', color='#16202a'),
    )
    return fig


def summary_report() -> dict[str, Any]:
    expl = STATE['explanation']
    comp = STATE['composition']
    comp_obj = comp['comp']
    edge_rows = edge_report(comp['edges'], comp['plan'].beta)
    return {
        'risk': STATE['risk'],
        'mode': STATE['mode'],
        'selected_class': expl['report']['selected_class'],
        'reduction_loss_delta': expl['report']['reduction_loss'],
        'uncertainty': expl['report']['uncertainty'],
        'composition': {
            'diagnostic': getattr(comp_obj, 'code', None),
            'gamma': edge_rows[0]['gamma'] if edge_rows else None,
            'loss': comp.get('loss'),
            'index': comp.get('index'),
        },
    }


def download_report() -> None:
    payload = summary_report()
    path = REPORTS / 'defense_demo_report.json'
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding='utf-8')
    ui.download(path.read_bytes(), 'defense_demo_report.json')


def controls(on_change) -> None:
    df = STATE['df']
    with ui.element('div').classes('fx-topbar w-full'):
        with ui.row().classes('fx-shell w-full items-center gap-3 p-3'):
            ui.label('FuzzyXAI: демонстрация глав 2-3').classes('text-lg fx-title')
            ui.label('Главы 2-3').classes('fx-chip')
            risk = ui.slider(min=0.0, max=1.0, value=STATE['risk'], step=0.01).props('label-always').classes('w-56')
            mode = ui.select(['audit', 'user'], value=STATE['mode'], label='режим').classes('w-36')
            feature = ui.select(feature_options(), value=STATE['feature'], label='признак').classes('w-48')
            conflict = ui.switch('показать конфликт', value=STATE['conflict'])

            def sync() -> None:
                STATE['risk'] = float(risk.value)
                STATE['mode'] = str(mode.value)
                STATE['feature'] = str(feature.value)
                STATE['conflict'] = bool(conflict.value)
                build_plan_from_state()
                recompute()
                feature.options = feature_options()
                feature.update()
                on_change()

            ui.button(icon='refresh', on_click=lambda: safe(sync, where='recompute')).props('flat round color=primary')
            ui.button(icon='download', on_click=download_report).props('flat round')
            ui.label(f'{len(df)} rows').classes('text-sm fx-muted ml-auto')


def plan_section() -> None:
    plan = STATE['plan']
    with ui.element('section').classes('fx-panel w-full'):
        ui.label('1. Из данных в термы').classes('text-lg fx-title fx-step')
        with ui.element('div').classes('fx-note w-full'):
            ui.label('Здесь числовой признак превращается в понятные термы: низкий, средний, высокий. Это вход для главы 2.').classes('text-sm')
        row_metrics([
            ('Строк', plan.metadata.get('n_rows'), 'demo-таблица'),
            ('Числовых признаков', len(plan.metadata.get('numeric_features', [])), ''),
            ('Сейчас показан', STATE['feature'], ''),
        ])
        ui.plotly(explainplan_membership_figure(plan, STATE['feature'], df=STATE['df'])).classes('w-full')
        rows = quantile_rows()
        if rows:
            ui.table(
                columns=[{'name': k, 'label': k, 'field': k} for k in ['term', 'shape', 'left', 'center', 'right']],
                rows=rows,
            ).classes('w-full q-table')


def explanation_section() -> None:
    expl = STATE['explanation']
    obj = expl['object']
    report = expl['report']
    with ui.element('section').classes('fx-panel w-full'):
        ui.label('2. Объяснение одного кейса').classes('text-lg fx-title fx-step')
        with ui.element('div').classes('fx-note w-full'):
            ui.label('Система выбирает класс нечёткого представления и строит объект объяснения: риск, правила, неопределённость, потеря редукции.').classes('text-sm')
        row_metrics([
            ('Представление', report['selected_class'], readable_class_name(report['selected_class'])),
            ('Неопределённость', round(obj.uncertainty, 4), 'меньше значит увереннее'),
            ('Потеря упрощения', round(obj.reduction_loss, 4), report['reduction_policy']),
        ])
        with ui.row().classes('w-full gap-3'):
            with ui.column().classes('w-full'):
                ui.plotly(membership_bar()).classes('w-full')
            with ui.column().classes('w-full'):
                ui.plotly(representation_figure(expl['representation'], title='Представление неопределенности')).classes('w-full')
        rules = [
            {
                'rule': r.name,
                'condition': ', '.join(f'{k}={v}' for k, v in r.conditions.items()),
                'activation': round(obj.activations.get(r.name, 0.0), 4),
                'result': r.conclusion,
            }
            for r in obj.rules
        ]
        ui.table(
            columns=[
                {'name': 'rule', 'label': 'правило', 'field': 'rule'},
                {'name': 'condition', 'label': 'условие', 'field': 'condition'},
                {'name': 'activation', 'label': 'активация', 'field': 'activation'},
                {'name': 'result', 'label': 'вывод', 'field': 'result'},
            ],
            rows=rules,
        ).classes('w-full')
        ui.label(expl['text']).classes('text-sm leading-relaxed')


def composition_section() -> None:
    comp = STATE['composition']
    comp_obj = comp['comp']
    rows = edge_report(comp['edges'], comp['plan'].beta)
    gamma = rows[0]['gamma'] if rows else 'n/a'
    diagnostic = getattr(comp_obj, 'code', None)
    with ui.element('section').classes('fx-panel w-full'):
        ui.label('3. Проверка цепочки: модель -> решение').classes('text-lg fx-title fx-step')
        with ui.element('div').classes('fx-note w-full'):
            ui.label('Это главный смысл главы 2: объяснение проверяется не только внутри модели, а между компонентами системы.').classes('text-sm')
        if diagnostic:
            with ui.element('div').classes('fx-status-bad w-full'):
                ui.label('Остановлено: компоненты используют несовместимые термы.').classes('text-base fx-title')
                ui.label('Так система показывает место, где объяснимость ломается, вместо красивого, но неверного отчёта.').classes('text-sm')
            row_metrics([
                ('Диагностика', diagnostic, getattr(comp_obj, 'reason', '')),
                ('Расхождение', gamma, '0 хорошо, 1 плохо'),
                ('Итог', 'остановлено', 'нужен аудит'),
            ])
        else:
            level, _, caption = agreement_level(float(gamma))
            with ui.element('div').classes('fx-status-ok w-full'):
                ui.label('Проверка пройдена: объяснения можно связать в цепочку.').classes('text-base fx-title')
                ui.label(f'Уровень расхождения: {level}; {caption}.').classes('text-sm')
            row_metrics([
                ('Расхождение', round(float(gamma), 4), '0 хорошо, 1 плохо'),
                ('Потеря', round(float(comp['loss']), 4), 'интерпретируемость'),
                ('Индекс', round(float(comp['index']), 4), 'чем выше, тем лучше'),
            ])
        ui.plotly(composition_story_figure(comp, rows)).classes('w-full')
        safe_rows = [{k: ', '.join(v) if isinstance(v, list) else v for k, v in row.items()} for row in rows]
        if safe_rows:
            ui.table(
                columns=[
                    {'name': 'source', 'label': 'откуда', 'field': 'source'},
                    {'name': 'target', 'label': 'куда', 'field': 'target'},
                    {'name': 'gamma', 'label': 'расхождение', 'field': 'gamma'},
                    {'name': 'severity', 'label': 'уровень', 'field': 'severity'},
                    {'name': 'left_terms', 'label': 'термы слева', 'field': 'left_terms'},
                    {'name': 'right_terms', 'label': 'термы справа', 'field': 'right_terms'},
                ],
                rows=safe_rows,
            ).classes('w-full')


def advanced_section() -> None:
    with ui.expansion('Технические детали и отчёт').classes('w-full'):
        profile = set(STATE['explanation']['profile'])
        candidates = default_candidates()
        selected = select_minimal_sufficient(profile, candidates, mode=STATE['mode'])
        front = {c.name for c in pareto_front([c for c in candidates if c.covers(profile)])}
        rows = [
            {
                'class': c.name,
                'covers': c.covers(profile),
                'pareto': c.name in front,
                'C_cog': c.cognitive_complexity,
                'C_comp': c.computational_complexity,
                'Delta_hat': c.expected_reduction_loss,
            }
            for c in candidates
        ]
        ui.label(f"Selected: {getattr(selected, 'name', getattr(selected, 'code', selected))}").classes('fx-title')
        ui.table(columns=[{'name': k, 'label': k, 'field': k} for k in rows[0].keys()], rows=rows).classes('w-full')
        ui.json_editor({'content': summary_report()}).classes('w-full h-72')


@ui.page('/')
def page() -> None:
    build_plan_from_state()
    recompute()
    apply_style()
    body = ui.column().classes('fx-shell w-full gap-4 pb-6')

    def redraw() -> None:
        body.clear()
        with body:
            controls(redraw)
            with ui.row().classes('w-full gap-4 items-start'):
                with ui.column().classes('w-full xl:w-[49%] gap-4'):
                    plan_section()
                with ui.column().classes('w-full xl:w-[49%] gap-4'):
                    explanation_section()
            composition_section()
            advanced_section()

    redraw()


def run() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--port', type=int, default=8085)
    args = parser.parse_args()
    ui.run(title='FuzzyXAI Defense Demo', port=args.port, reload=False, show=True)


if __name__ in {'__main__', '__mp_main__'}:
    run()
