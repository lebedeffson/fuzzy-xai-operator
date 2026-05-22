"""NiceGUI local dashboard for FuzzyXAI doctoral demonstration.

This v8 GUI is synthetic-first: every page works out of the box without an
uploaded dataset or pre-generated reports. Uploaded CSV files are optional.

Run from repository root:
    python apps/nicegui_dashboard.py --port 8080

Optional desktop-like native window if pywebview is installed:
    python apps/nicegui_dashboard.py --native
"""
from __future__ import annotations

import argparse
import io
import json
import sys
import traceback
from pathlib import Path
from typing import Any, Dict, Mapping, Sequence

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

try:
    import pandas as pd
    import plotly.graph_objects as go
    from nicegui import ui
except Exception as exc:  # pragma: no cover
    raise SystemExit('NiceGUI dashboard requires nicegui, pandas and plotly. Install requirements.txt first.') from exc

from fuzzyxai import ExplainPlan, build_profile, interpretability_index, interpretability_loss, pareto_front, select_minimal_sufficient
from fuzzyxai.core.plan_builder import build_explain_plan_from_dataframe
from fuzzyxai.reporting.session_report import SessionReport
from fuzzyxai.selection.compatibility import compatible_types, synthesize_levels
from fuzzyxai.visual.composition_graph import edge_report
from fuzzyxai.visual.interactive_graph import composition_plotly_figure
from fuzzyxai.visual.representation_plots import explainplan_membership_figure, representation_figure
from fuzzyxai.demo.synthetic import (
    build_demo_composition,
    build_demo_explanation,
    default_candidates,
    default_plan,
    metadata_for_demo,
    sample_dataframe,
)

REPORTS = ROOT / 'reports'
ASSETS = ROOT / 'apps' / 'assets'
SESSION_LOG = REPORTS / 'nicegui_session_log.csv'
REPORTS.mkdir(exist_ok=True)

STATE: Dict[str, Any] = {
    'plan': default_plan(),
    'df': sample_dataframe(),
    'session_report': SessionReport('FuzzyXAI NiceGUI local session', metadata={'frontend': 'NiceGUI', 'version': 'v8-gui-fixed'}, log_path=SESSION_LOG),
    'last_explanation': None,
    'composition': None,
    'fml_profile': ['u_num', 'u_ling', 'u_int', 'u_exp', 'u_conf', 'u_trace'],
    'fml_levels': [['u_int', 'u_num'], ['u_exp'], ['u_conf', 'u_ling', 'u_trace']],
    'dark_mode': False,
}


def load_style() -> None:
    css = ASSETS / 'nicegui_style.css'
    if css.exists():
        ui.add_head_html(f'<style>{css.read_text(encoding="utf-8")}</style>')


def add_step(name: str, payload: Mapping[str, Any]) -> None:
    try:
        STATE['session_report'].add_step(name, payload)
    except Exception:
        pass


def notify_ok(message: str) -> None:
    ui.notify(message, type='positive', position='top-right')


def notify_warn(message: str) -> None:
    ui.notify(message, type='warning', position='top-right')


def safe_call(name: str, fn):
    """Run GUI handler with visible errors instead of blank panels."""
    try:
        return fn()
    except Exception as exc:  # pragma: no cover - GUI safety net
        add_step('gui_error', {'where': name, 'error': str(exc), 'traceback': traceback.format_exc()[-2000:]})
        ui.notify(f'Ошибка в {name}: {exc}', type='negative', position='top-right', timeout=7000)
        return None


def ensure_demo_state() -> None:
    if STATE.get('df') is None:
        STATE['df'] = sample_dataframe()
    if STATE.get('plan') is None:
        STATE['plan'] = default_plan()
    if STATE.get('last_explanation') is None:
        STATE['last_explanation'] = build_demo_explanation(0.72, plan=STATE['plan'])
    if STATE.get('composition') is None:
        STATE['composition'] = build_demo_composition(0.72, 0.74, plan=STATE['plan'], conflict=False)


def download_json(filename: str, payload: Mapping[str, Any]) -> None:
    content = json.dumps(payload, ensure_ascii=False, indent=2)
    try:
        ui.download(content.encode('utf-8'), filename)
    except TypeError:
        ui.download(content, filename=filename)


def metric_row(items: Sequence[tuple[str, Any, str | None]]) -> None:
    with ui.row().classes('w-full gap-3'):
        for title, value, caption in items:
            with ui.card().classes('fx-card p-4 min-w-[160px] grow'):
                ui.label(str(title)).classes('text-sm fx-muted')
                ui.label(str(value)).classes('text-2xl fx-title')
                if caption:
                    ui.label(caption).classes('text-xs fx-muted')


def explain_plan_to_dict(plan: ExplainPlan) -> Dict[str, Any]:
    return {
        'beta': dict(plan.beta),
        'lambda': dict(plan.lambda_),
        'eta': dict(plan.eta),
        'i_min': plan.i_min,
        'activation_threshold': plan.activation_threshold,
        'epsilon': plan.epsilon,
        'metadata': dict(plan.metadata),
    }


def membership_term_figure(plan: ExplainPlan, feature: str | None = None):
    return explainplan_membership_figure(plan, feature, df=STATE.get('df'))


def home_panel() -> None:
    ensure_demo_state()
    with ui.column().classes('w-full gap-4'):
        with ui.element('div').classes('fx-hero w-full'):
            ui.label('FuzzyXAI Local GUI v8').classes('text-4xl fx-title')
            ui.label('Стабильная синтетическая демонстрация для глав 2 и 3: всё работает без CSV, без внешних отчётов и без облака.').classes('text-lg')
            with ui.row().classes('gap-2 mt-4'):
                ui.label('Chapter 2: operator').classes('fx-chip')
                ui.label('Chapter 3: hierarchy').classes('fx-chip')
                ui.label('synthetic-first').classes('fx-chip')
        metric_row([
            ('Data', f"{len(STATE['df'])} rows", 'built-in synthetic medical table'),
            ('Plan', 'ready', 'ExplainPlan is initialized at startup'),
            ('Composition', 'ready', 'default graph is generated'),
            ('Reports', 'fallback', 'missing files are handled gracefully'),
        ])
        with ui.card().classes('fx-card w-full p-4'):
            ui.label('Быстрый маршрут (3 шага)').classes('text-xl fx-title')
            ui.markdown('''
1. Открой **1. План (ExplainPlan)** и нажми кнопку построения.  
2. Открой **2. Объяснение кейса** и посмотри `A_k^F`, правила и текст.  
3. Открой **3. Граф согласования** и проверь конфликт/согласование (`D_ij`, `γ`, `I`).
''')


def explainplan_panel() -> None:
    ensure_demo_state()
    with ui.column().classes('w-full gap-4'):
        ui.label('1. План (ExplainPlan)').classes('text-3xl fx-title')
        ui.label('Шаг 1: выбери данные. Шаг 2: нажми "Построить план". Шаг 3: смотри график и интерпретацию.').classes('fx-muted')
        with ui.row().classes('w-full gap-4'):
            with ui.card().classes('fx-card p-4 w-[380px]'):
                ui.label('Управление').classes('text-xl fx-title')
                status = ui.label('').classes('fx-muted')
                target_select = ui.select([], label='Целевая колонка (target)').classes('w-full')
                feature_select = ui.select([], label='Признак для графика').classes('w-full')
                n_terms = ui.slider(min=3, max=5, value=3, step=1).props('label-always').classes('w-full')
                mode_select = ui.select(['user', 'audit'], value='audit', label='Режим').classes('w-full')

                def refresh_selects(df: pd.DataFrame) -> None:
                    columns = list(df.columns)
                    target_select.options = columns
                    target_select.value = 'target' if 'target' in columns else (columns[-1] if columns else None)
                    metadata = (STATE.get('plan').metadata if STATE.get('plan') is not None else {}) or {}
                    numeric_first = [c for c in metadata.get('numeric_features', []) if c in columns and c != target_select.value]
                    rest = [c for c in columns if c != target_select.value and c not in numeric_first]
                    feature_select.options = numeric_first + rest
                    feature_select.value = feature_select.options[0] if feature_select.options else None
                    target_select.update(); feature_select.update()
                    status.text = f'Загружено: {len(df)} строк, {len(df.columns)} колонок.'
                    status.update()

                def load_sample() -> None:
                    def _run():
                        STATE['df'] = sample_dataframe()
                        refresh_selects(STATE['df'])
                        STATE['plan'] = build_explain_plan_from_dataframe(STATE['df'], target='target', n_terms=int(n_terms.value), mode=mode_select.value).with_reduction_weight(0.10)
                        add_step('load_sample_data', {'rows': len(STATE['df']), 'columns': list(STATE['df'].columns)})
                        redraw_plan()
                        notify_ok('Синтетические данные загружены, ExplainPlan построен')
                    safe_call('load_sample_data', _run)

                def on_upload(e) -> None:
                    def _run():
                        content = e.content.read() if hasattr(e.content, 'read') else e.content
                        df = pd.read_csv(io.BytesIO(content))
                        STATE['df'] = df
                        refresh_selects(df)
                        notify_ok(f'CSV загружен: {getattr(e, "name", "file.csv")}')
                        add_step('upload_csv', {'rows': len(df), 'columns': list(df.columns)})
                    safe_call('upload_csv', _run)

                def build_plan() -> None:
                    def _run():
                        df = STATE.get('df')
                        if df is None or len(df) == 0:
                            df = sample_dataframe()
                            STATE['df'] = df
                            refresh_selects(df)
                        target = target_select.value if target_select.value else ('target' if 'target' in df.columns else None)
                        STATE['plan'] = build_explain_plan_from_dataframe(df, target=target, n_terms=int(n_terms.value), mode=mode_select.value).with_reduction_weight(0.10)
                        add_step('build_explain_plan', explain_plan_to_dict(STATE['plan']))
                        redraw_plan()
                        notify_ok('ExplainPlan построен и передан в ядро')
                    safe_call('build_explain_plan', _run)

                ui.button('Взять demo-данные и построить план', on_click=load_sample).props('color=primary').classes('w-full')
                ui.upload(label='Загрузить CSV (опционально)', auto_upload=True, on_upload=on_upload).classes('w-full')
                ui.button('Построить / обновить ExplainPlan', on_click=build_plan).classes('w-full')
                ui.button('Скачать ExplainPlan (JSON)', on_click=lambda: download_json('explain_plan.json', explain_plan_to_dict(STATE['plan']))).classes('w-full')

            with ui.card().classes('fx-card p-4 grow'):
                ui.label('Что получилось').classes('text-xl fx-title')
                plot_container = ui.column().classes('w-full')
                meta_container = ui.column().classes('w-full')

        def redraw_plan() -> None:
            plot_container.clear(); meta_container.clear()
            plan = STATE['plan']
            with plot_container:
                ui.plotly(membership_term_figure(plan, feature_select.value)).classes('w-full')
                ui.markdown(
                    '**Что показывает график:**\n'
                    '- Для числовых признаков: кривые "низкий / средний / высокий".\n'
                    '- Для категориальных: столбики частот категорий.\n'
                    '- По оси Y всегда степень принадлежности (0..1) или частота.'
                ).classes('text-sm')
            with meta_container:
                df = STATE.get('df')
                if df is None:
                    df = sample_dataframe()
                ui.table(columns=[{'name': c, 'label': c, 'field': c} for c in df.columns], rows=df.head(6).to_dict('records')).classes('w-full')
                metadata = plan.metadata or {}
                numeric = metadata.get('numeric_features', [])
                categorical = metadata.get('categorical_features', [])
                metric_row([
                    ('Rows', metadata.get('n_rows', len(df)), None),
                    ('Numeric', len(numeric), ', '.join(numeric[:3]) if numeric else None),
                    ('Categorical', len(categorical), ', '.join(categorical[:3]) if categorical else None),
                    ('β reduction', round(plan.beta.get('reduction', 0.0), 3), 'extended d_E'),
                ])
                ui.json_editor({'content': explain_plan_to_dict(plan)}).classes('w-full h-[380px]')

        refresh_selects(STATE['df'])
        redraw_plan()


def explain_instance_panel() -> None:
    ensure_demo_state()
    with ui.column().classes('w-full gap-4'):
        ui.label('2. Объяснение кейса').classes('text-3xl fx-title')
        ui.label('Нажми одну кнопку: система построит E_k, выберет класс F* и покажет человеческое объяснение.').classes('fx-muted')
        with ui.row().classes('w-full gap-4'):
            with ui.card().classes('fx-card p-4 w-[380px]'):
                risk = ui.slider(min=0.0, max=1.0, value=0.72, step=0.01).props('label-always').classes('w-full')
                ui.label('Метаданные ситуации').classes('text-lg fx-title')
                has_intervals = ui.checkbox('Интервальные измерения', value=True)
                num_experts = ui.number('Число экспертов', value=2, min=1, max=10, step=1).classes('w-full')
                source_conflict = ui.checkbox('Конфликт источников', value=True)
                requires_audit = ui.checkbox('Требуется аудит', value=True)
                cf_instability = ui.checkbox('Контрфактическая нестабильность', value=False)
                audience = ui.select(['doctor', 'patient', 'engineer'], value='doctor', label='Аудитория').classes('w-full')
                use_llm = ui.checkbox('LLM-текст, если задан OPENAI_API_KEY', value=False)
                auto_refresh = ui.checkbox('Автопересчёт при движении риска', value=True)

                def run_explanation() -> None:
                    def _run():
                        metadata = metadata_for_demo(
                            intervals=bool(has_intervals.value),
                            experts=int(num_experts.value or 1),
                            conflict=bool(source_conflict.value),
                            audit=bool(requires_audit.value),
                            cf=bool(cf_instability.value),
                        )
                        last = build_demo_explanation(float(risk.value), plan=STATE['plan'], metadata=metadata, audience=audience.value, use_llm=bool(use_llm.value))
                        STATE['last_explanation'] = last
                        add_step('explain_instance', last['report'])
                        redraw_instance()
                    safe_call('explain_instance', _run)

                ui.button('Build / refresh explanation', on_click=run_explanation).props('color=primary').classes('w-full')
                risk.on_value_change(lambda e: run_explanation() if auto_refresh.value else None)
                ui.button('Download E_k report JSON', on_click=lambda: download_json('explanation_object_report.json', STATE.get('last_explanation', {}).get('report', {}))).classes('w-full')

            result_container = ui.column().classes('grow gap-4')

        def redraw_instance() -> None:
            result_container.clear()
            last = STATE.get('last_explanation') or build_demo_explanation(0.72, plan=STATE['plan'])
            STATE['last_explanation'] = last
            obj = last['object']; report = last['report']; representation = last['representation']; profile = last['profile']
            with result_container:
                metric_row([
                    ('Selected', report.get('selected_class'), 'Парето-выбор'),
                    ('u(E_k)', round(obj.uncertainty, 4), 'агрегированная неопределённость'),
                    ('Δ_k', round(obj.reduction_loss, 4), f"policy={report.get('reduction_policy')}"),
                    ('Profile', len(profile), ', '.join(sorted(profile))),
                ])
                with ui.tabs().classes('w-full') as tabs:
                    t1 = ui.tab('Membership')
                    t2 = ui.tab('A_k^F')
                    t3 = ui.tab('Rules')
                    t4 = ui.tab('Text')
                    t5 = ui.tab('JSON')
                with ui.tab_panels(tabs, value=t1).classes('w-full'):
                    with ui.tab_panel(t1):
                        memberships = obj.metadata.get('memberships', {})
                        fig = go.Figure(go.Bar(x=list(memberships.keys()), y=list(memberships.values()), marker_color=['#22c55e', '#f59e0b', '#ef4444']))
                        fig.update_layout(title='Linguistic memberships', yaxis=dict(range=[0, 1]), height=330)
                        ui.plotly(fig).classes('w-full')
                    with ui.tab_panel(t2):
                        ui.plotly(representation_figure(representation, title='A_k^F: multilevel uncertainty representation')).classes('w-full')
                        ui.markdown(
                            '**Что это:** визуализация выбранного нечёткого представления `A_k^F`.\n'
                            'Интервалы показывают диапазон, hesitant — набор экспертных значений, neutrosophic — T/I/F.'
                        ).classes('text-sm')
                    with ui.tab_panel(t3):
                        rows = [{'rule': r.name, 'condition': str(dict(r.conditions)), 'conclusion': r.conclusion, 'activation': round(obj.activations.get(r.name, 0.0), 4)} for r in obj.rules]
                        ui.table(columns=[{'name': k, 'label': k, 'field': k} for k in ['rule', 'condition', 'conclusion', 'activation']], rows=rows).classes('w-full')
                    with ui.tab_panel(t4):
                        ui.label(last['text']).classes('text-base')
                        ui.label(f"Режим генерации: {report.get('text_generation', {}).get('mode', 'template')}").classes('text-xs fx-muted')
                    with ui.tab_panel(t5):
                        ui.json_editor({'content': report}).classes('w-full h-[440px]')

        redraw_instance()


def composition_panel() -> None:
    ensure_demo_state()
    with ui.column().classes('w-full gap-4'):
        ui.label('3. Граф согласования').classes('text-3xl fx-title')
        ui.label('Граф показывает, где объяснения согласованы, а где конфликтуют. Краснее = больше рассогласование.').classes('fx-muted')
        with ui.row().classes('w-full gap-4'):
            with ui.card().classes('fx-card p-4 w-[380px]'):
                risk_a = ui.slider(min=0.0, max=1.0, value=0.72, step=0.01).props('label-always').classes('w-full')
                risk_b = ui.slider(min=0.0, max=1.0, value=0.74, step=0.01).props('label-always').classes('w-full')
                conflict = ui.switch('Сымитировать разрыв терминов D_ij', value=False)
                beta_delta = ui.slider(min=0.0, max=0.3, value=float(STATE['plan'].beta.get('reduction', 0.10)), step=0.01).props('label-always').classes('w-full')

                def run_composition() -> None:
                    def _run():
                        plan = STATE['plan'].with_reduction_weight(float(beta_delta.value))
                        data = build_demo_composition(float(risk_a.value), float(risk_b.value), plan=plan, conflict=bool(conflict.value))
                        STATE['composition'] = data
                        add_step('compose', {'risk_a': risk_a.value, 'risk_b': risk_b.value, 'conflict': bool(conflict.value)})
                        redraw_composition()
                        notify_ok('Композиция рассчитана')
                    safe_call('compose', _run)

                def export_graph() -> None:
                    data = STATE.get('composition') or build_demo_composition(plan=STATE['plan'])
                    html = composition_plotly_figure(data['edges'], data['plan'].beta).to_html(include_plotlyjs='cdn')
                    try:
                        ui.download(html.encode('utf-8'), 'composition_graph.html')
                    except TypeError:
                        ui.download(html, filename='composition_graph.html')

                ui.button('Recalculate composition', on_click=run_composition).props('color=primary').classes('w-full')
                ui.button('Download graph HTML', on_click=export_graph).classes('w-full')
                conflict.on_value_change(lambda e: run_composition())
            graph_container = ui.column().classes('grow gap-4')

        def redraw_composition() -> None:
            graph_container.clear()
            data = STATE.get('composition') or build_demo_composition(plan=STATE['plan'])
            STATE['composition'] = data
            comp = data['comp']; edges = data['edges']; plan = data['plan']
            with graph_container:
                ui.plotly(composition_plotly_figure(edges, plan.beta)).classes('w-full')
                rows = edge_report(edges, plan.beta)
                if hasattr(comp, 'code'):
                    metric_row([
                        ('Diagnostic', comp.code, comp.reason),
                        ('Severity', comp.severity, 'composition rejected'),
                        ('γ', round(rows[0]['gamma'], 4) if rows else 'n/a', 'edge disagreement'),
                    ])
                    ui.label('Разрыв не скрывается: система формирует диагностическое состояние, а не фальшивое объяснение.').classes('text-red-600')
                else:
                    loss = data.get('loss')
                    index = data.get('index')
                    metric_row([
                        ('γ', round(comp.metadata.get('gamma', 0.0), 4), 'semantic disagreement'),
                        ('u_ij', round(comp.uncertainty, 4), 'probabilistic t-conorm'),
                        ('L_ext', round(loss, 4), 'loss'),
                        ('I(E_G)', round(index, 4), 'index'),
                    ])
                if rows:
                    safe_rows = []
                    for row in rows:
                        safe_rows.append({k: (', '.join(v) if isinstance(v, list) else v) for k, v in row.items()})
                    ui.table(columns=[{'name': k, 'label': k, 'field': k} for k in safe_rows[0].keys()], rows=safe_rows).classes('w-full')

        redraw_composition()


def about_panel() -> None:
    with ui.column().classes('w-full gap-4'):
        ui.label('О системе').classes('text-3xl fx-title')
        ui.markdown(
            '### Глава 2. Системный оператор\n'
            'Связывает объяснения разных блоков ИИ в единую цепочку и считает:\n'
            '- рассогласование `d_E`\n'
            '- потерю интерпретируемости `L`\n'
            '- итоговый индекс `I(E_G)`\n\n'
            '### Глава 3. Иерархия нечётких представлений\n'
            'Подбирает минимально достаточный класс для текущей ситуации:\n'
            '- `F0` — базовая нечёткость\n'
            '- `FI` — интервалы\n'
            '- `FH` — экспертная разноголосица\n'
            '- `FNsrc` — конфликт источников\n'
            '- `FML` — многоуровневая композиция\n\n'
            '### Как пользоваться GUI\n'
            '1. ExplainPlan: построить термы.\n'
            '2. Explain Instance: получить объяснение для кейса.\n'
            '3. Composition Graph: увидеть конфликт между модулями.\n'
            '4. Thesis Demo: сформировать отчёты для защиты.'
        )

def synthesizer_panel() -> None:
    ensure_demo_state()
    all_types = ['u_num', 'u_ling', 'u_int', 'u_exp', 'u_conf', 'u_trace', 'u_cf', 'u_shift', 'u_user', 'u_rule', 'u_time']
    with ui.column().classes('w-full gap-4'):
        ui.label('4. Конструктор FML (расширенный режим)').classes('text-3xl fx-title')
        ui.label('Выбранные типы хранятся в состоянии GUI. Можно переключаться между вкладками, настройки не теряются.').classes('fx-muted')
        with ui.row().classes('w-full gap-4'):
            with ui.card().classes('fx-card p-4 w-[380px]'):
                selected_types = ui.select(all_types, multiple=True, value=STATE['fml_profile'], label='Profile types').classes('w-full')
                ui.label('Manual levels').classes('text-lg fx-title')
                levels_state = STATE.get('fml_levels') or [[], [], []]
                level1 = ui.select(all_types, multiple=True, value=levels_state[0] if len(levels_state) > 0 else [], label='Level 1').classes('w-full')
                level2 = ui.select(all_types, multiple=True, value=levels_state[1] if len(levels_state) > 1 else [], label='Level 2').classes('w-full')
                level3 = ui.select(all_types, multiple=True, value=levels_state[2] if len(levels_state) > 2 else [], label='Level 3').classes('w-full')

                def save_state() -> None:
                    STATE['fml_profile'] = list(selected_types.value or [])
                    STATE['fml_levels'] = [list(level1.value or []), list(level2.value or []), list(level3.value or [])]

                def auto_synthesize() -> None:
                    def _run():
                        save_state()
                        levels = synthesize_levels(set(STATE['fml_profile']))
                        vals = [sorted(x) for x in levels]
                        level1.value = vals[0] if len(vals) > 0 else []
                        level2.value = vals[1] if len(vals) > 1 else []
                        level3.value = vals[2] if len(vals) > 2 else []
                        level1.update(); level2.update(); level3.update()
                        save_state()
                        redraw_synth()
                        notify_ok('Уровни синтезированы жадным алгоритмом')
                    safe_call('auto_synthesize', _run)

                def reset_synth() -> None:
                    STATE['fml_profile'] = ['u_num', 'u_ling', 'u_int', 'u_exp', 'u_conf', 'u_trace']
                    STATE['fml_levels'] = [['u_int', 'u_num'], ['u_exp'], ['u_conf', 'u_ling', 'u_trace']]
                    selected_types.value = STATE['fml_profile']; level1.value, level2.value, level3.value = STATE['fml_levels']
                    selected_types.update(); level1.update(); level2.update(); level3.update()
                    redraw_synth()

                ui.button('Auto-synthesize F_ML', on_click=auto_synthesize).props('color=primary').classes('w-full')
                ui.button('Apply / check configuration', on_click=lambda: (save_state(), redraw_synth())).classes('w-full')
                ui.button('Reset synthetic profile', on_click=reset_synth).classes('w-full')
            synth_container = ui.column().classes('grow gap-4')

        def redraw_synth() -> None:
            synth_container.clear()
            save_state()
            profile = set(STATE['fml_profile'])
            levels = [set(x) for x in STATE['fml_levels'] if x]
            compatibility_rows = [{'level': idx, 'types': ', '.join(sorted(lvl)), 'compatible': compatible_types(lvl)} for idx, lvl in enumerate(levels, 1)]
            candidates = default_candidates()
            selected = select_minimal_sufficient(profile, candidates, mode='audit')
            admissible = [c for c in candidates if c.covers(profile)]
            front = pareto_front(admissible) if admissible else []
            with synth_container:
                metric_row([
                    ('Profile size', len(profile), ', '.join(sorted(profile))),
                    ('Levels', len(levels), 'manual or greedy'),
                    ('Selected', getattr(selected, 'name', getattr(selected, 'code', 'none')), 'candidate'),
                    ('Covered', 'yes' if not hasattr(selected, 'code') else 'no', 'D_choice if no'),
                ])
                ui.table(columns=[{'name': k, 'label': k, 'field': k} for k in ['level', 'types', 'compatible']], rows=compatibility_rows).classes('w-full')
                rows = [{
                    'name': c.name,
                    'covers': c.covers(profile),
                    'C_cog': c.cognitive_complexity,
                    'C_comp': c.computational_complexity,
                    'Delta_hat': c.expected_reduction_loss,
                    'pareto': c.name in {f.name for f in front},
                } for c in candidates]
                ui.table(columns=[{'name': k, 'label': k, 'field': k} for k in rows[0].keys()], rows=rows).classes('w-full')
                if hasattr(selected, 'code'):
                    ui.label(f'D_choice: {selected.context}').classes('text-red-600')
                add_step('synthesize_fml', {'profile': sorted(profile), 'levels': [sorted(l) for l in levels], 'selected': getattr(selected, 'name', getattr(selected, 'code', None))})

        redraw_synth()


def report_file_card(title: str, filename: str) -> None:
    path = REPORTS / filename
    with ui.card().classes('fx-card p-4 grow'):
        ui.label(title).classes('text-xl fx-title')
        if path.exists():
            try:
                data = json.loads(path.read_text(encoding='utf-8'))
                ui.button(f'Download {filename}', on_click=lambda p=path, f=filename: ui.download(p.read_bytes(), f)).classes('mb-2')
                ui.json_editor({'content': data}).classes('w-full h-[360px]')
            except Exception as exc:
                ui.label(f'Файл есть, но не прочитан: {exc}').classes('text-red-600')
        else:
            ui.label(f'Отчёт {filename} пока не создан. Откройте Thesis Demo и запустите проверку.').classes('fx-muted')


def reports_panel() -> None:
    with ui.column().classes('w-full gap-4'):
        ui.label('5. Отчёты').classes('text-3xl fx-title')
        ui.label('Если отчёта нет, страница не падает: она показывает, что нужно запустить Thesis Demo.').classes('fx-muted')
        with ui.row().classes('w-full gap-4'):
            report_file_card('Thesis validation', 'thesis_validation.json')
            report_file_card('Thesis demo', 'thesis_demo_report.json')
        with ui.row().classes('w-full gap-4'):
            report_file_card('Calibration', 'chapter2_calibration_report.json')
            report_file_card('Benchmark', 'breast_cancer_benchmark.json')


def thesis_demo_panel() -> None:
    with ui.column().classes('w-full gap-4'):
        ui.label('6. Thesis Demo (для защиты)').classes('text-3xl fx-title')
        ui.label('Одна кнопка для числовой валидации и одна кнопка для полного маршрута глав 2–3.').classes('fx-muted')
        with ui.row().classes('w-full gap-4'):
            with ui.card().classes('fx-card p-4 grow'):
                ui.label('1. Numerical validation').classes('text-xl fx-title')
                validation_box = ui.column().classes('w-full')

                def run_validation() -> None:
                    def _run():
                        validation_box.clear()
                        from proofs.validate_thesis_examples import build_validation_report
                        report = build_validation_report()
                        (REPORTS / 'thesis_validation.json').write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
                        add_step('thesis_validation', {'status': report['status'], 'total_checks': report['total_checks'], 'failed_checks': report['failed_checks']})
                        with validation_box:
                            metric_row([
                                ('Status', report['status'], 'PASS means thesis numbers match code'),
                                ('Checks', report['total_checks'], 'chapter 2 + chapter 3'),
                                ('Failed', report['failed_checks'], 'must be 0'),
                            ])
                            rows = report['chapters']['chapter2']['checks'] + report['chapters']['chapter3']['checks']
                            ui.table(columns=[
                                {'name': 'name', 'label': 'check', 'field': 'name'},
                                {'name': 'expected', 'label': 'expected', 'field': 'expected'},
                                {'name': 'actual', 'label': 'actual', 'field': 'actual'},
                                {'name': 'passed', 'label': 'passed', 'field': 'passed'},
                            ], rows=rows).classes('w-full')
                            ui.button('Download thesis_validation.json', on_click=lambda: ui.download((REPORTS / 'thesis_validation.json').read_bytes(), 'thesis_validation.json'))
                    safe_call('run_validation', _run)

                ui.button('Run thesis validation', on_click=run_validation).props('color=primary')

            with ui.card().classes('fx-card p-4 grow'):
                ui.label('2. Full thesis demo').classes('text-xl fx-title')
                demo_box = ui.column().classes('w-full')

                def run_thesis_demo() -> None:
                    def _run():
                        demo_box.clear()
                        from examples.thesis_demo import run_demo, write_markdown
                        report = run_demo()
                        json_path = REPORTS / 'thesis_demo_report.json'
                        md_path = REPORTS / 'thesis_demo_report.md'
                        json_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
                        write_markdown(report, md_path)
                        add_step('thesis_demo', {'status': report['status'], 'steps': len(report.get('timeline', [])), 'route': report.get('route')})
                        with demo_box:
                            metric_row([
                                ('Status', report['status'], 'end-to-end route'),
                                ('Steps', len(report.get('timeline', [])), 'demo timeline'),
                                ('Graph', 'HTML', report.get('artifacts', {}).get('graph_html')),
                            ])
                            for step in report.get('timeline', []):
                                with ui.expansion(f"{step['step']}. {step['name']}").classes('w-full'):
                                    ui.json_editor({'content': step.get('payload', {})}).classes('w-full h-[220px]')
                            ui.button('Download thesis_demo_report.json', on_click=lambda: ui.download(json_path.read_bytes(), 'thesis_demo_report.json'))
                    safe_call('run_thesis_demo', _run)

                ui.button('Run full thesis demo', on_click=run_thesis_demo).props('color=primary')


def session_panel() -> None:
    with ui.column().classes('w-full gap-4'):
        ui.label('7. Экспорт сессии').classes('text-3xl fx-title')
        ui.label('Все действия пишутся в JSON и CSV: это воспроизводимый след GUI-сессии.').classes('fx-muted')
        report_json = STATE['session_report'].to_json()
        ui.button('Download session_report.json', on_click=lambda: ui.download(report_json.encode('utf-8'), 'fuzzyxai_session_report.json')).props('color=primary')
        if SESSION_LOG.exists():
            ui.button('Download session_log.csv', on_click=lambda: ui.download(SESSION_LOG.read_bytes(), 'nicegui_session_log.csv'))
            ui.label(f'CSV audit log: {SESSION_LOG}').classes('text-xs fx-muted')
        ui.json_editor({'content': STATE['session_report'].to_dict()}).classes('w-full h-[620px]')


def advanced_panel() -> None:
    with ui.column().classes('w-full gap-4'):
        ui.label('Дополнительно').classes('text-3xl fx-title')
        ui.label('Здесь всё второстепенное: теория, отчёты для защиты и экспорт сессии.').classes('fx-muted')
        with ui.card().classes('fx-card w-full p-4'):
            ui.markdown(
                '**Коротко о системе:**\n'
                '- Глава 2: оператор согласования объяснений (`d_E`, `L`, `I`).\n'
                '- Глава 3: выбор нечёткого представления (`F0/FI/FH/FNsrc/FML`).\n'
                '- Цель GUI: показать это на одном кейсе без ручного кода.'
            )

        with ui.card().classes('fx-card w-full p-4'):
            ui.label('Отчёты для защиты').classes('text-xl fx-title')
            validation_box = ui.column().classes('w-full')
            demo_box = ui.column().classes('w-full')

            def run_validation() -> None:
                def _run():
                    validation_box.clear()
                    from proofs.validate_thesis_examples import build_validation_report
                    report = build_validation_report()
                    (REPORTS / 'thesis_validation.json').write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
                    with validation_box:
                        metric_row([
                            ('Status', report['status'], None),
                            ('Checks', report['total_checks'], None),
                            ('Failed', report['failed_checks'], None),
                        ])
                safe_call('advanced_validation', _run)

            def run_thesis_demo() -> None:
                def _run():
                    demo_box.clear()
                    from examples.thesis_demo import run_demo, write_markdown
                    report = run_demo()
                    json_path = REPORTS / 'thesis_demo_report.json'
                    md_path = REPORTS / 'thesis_demo_report.md'
                    json_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
                    write_markdown(report, md_path)
                    with demo_box:
                        metric_row([
                            ('Status', report['status'], None),
                            ('Steps', len(report.get('timeline', [])), None),
                            ('I(E_G)', report.get('summary', {}).get('index_I_ext', 'n/a'), None),
                        ])
                safe_call('advanced_thesis_demo', _run)

            with ui.row().classes('w-full gap-3'):
                ui.button('Запустить числовую валидацию', on_click=run_validation).props('color=primary')
                ui.button('Запустить полный thesis demo', on_click=run_thesis_demo)
                ui.button('Скачать thesis_validation.json', on_click=lambda: ui.download((REPORTS / 'thesis_validation.json').read_bytes(), 'thesis_validation.json') if (REPORTS / 'thesis_validation.json').exists() else notify_warn('Сначала запусти валидацию'))
                ui.button('Скачать thesis_demo_report.json', on_click=lambda: ui.download((REPORTS / 'thesis_demo_report.json').read_bytes(), 'thesis_demo_report.json') if (REPORTS / 'thesis_demo_report.json').exists() else notify_warn('Сначала запусти thesis demo'))

        with ui.card().classes('fx-card w-full p-4'):
            ui.label('Экспорт сессии').classes('text-xl fx-title')
            ui.button('Скачать session_report.json', on_click=lambda: ui.download(STATE['session_report'].to_json().encode('utf-8'), 'fuzzyxai_session_report.json')).props('color=primary')
            if SESSION_LOG.exists():
                ui.button('Скачать session_log.csv', on_click=lambda: ui.download(SESSION_LOG.read_bytes(), 'nicegui_session_log.csv'))


@ui.page('/')
def main_page() -> None:
    ensure_demo_state()
    load_style()
    ui.query('body').classes('bg-slate-50')
    dark = ui.dark_mode(value=False)
    with ui.header().classes('items-center justify-between bg-slate-950 text-white'):
        ui.label('FuzzyXAI Doctoral GUI v8').classes('text-xl fx-title')
        with ui.row().classes('items-center gap-3'):
            ui.label('local synthetic-first NiceGUI dashboard').classes('text-sm opacity-70')
            def toggle_dark(e):
                STATE['dark_mode'] = bool(e.value)
                dark.enable() if e.value else dark.disable()
                add_step('toggle_theme', {'dark_mode': bool(e.value)})
            ui.switch('Dark', value=False, on_change=toggle_dark).props('dense color=amber')
    with ui.left_drawer(value=True).classes('bg-white'):
        ui.label('Навигация').classes('text-lg fx-title p-4')
        ui.separator()
        tabs = ui.tabs().props('vertical').classes('w-full')
        with tabs:
            t_home = ui.tab('Старт')
            t_plan = ui.tab('1. План')
            t_instance = ui.tab('2. Объяснение')
            t_graph = ui.tab('3. Граф')
            t_advanced = ui.tab('Дополнительно')
    with ui.row().classes('w-full p-6'):
        with ui.tab_panels(tabs, value=t_home).classes('w-full'):
            with ui.tab_panel(t_home): home_panel()
            with ui.tab_panel(t_plan): explainplan_panel()
            with ui.tab_panel(t_instance): explain_instance_panel()
            with ui.tab_panel(t_graph): composition_panel()
            with ui.tab_panel(t_advanced): advanced_panel()


def run() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--native', action='store_true', help='open as native desktop window if pywebview is installed')
    parser.add_argument('--port', type=int, default=8080)
    args = parser.parse_args()
    ui.run(title='FuzzyXAI Doctoral GUI', port=args.port, reload=False, native=args.native, show=not args.native)


if __name__ in {'__main__', '__mp_main__'}:
    run()
