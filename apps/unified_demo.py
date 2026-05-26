"""Unified Defense Hub: one GUI for all dissertation artifacts and live demos."""
from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from apps.chapter5_web_demo import build_backend, evaluate_vector
from apps.services.hub_data import HubReports, load_reports
from fuzzyxai.data import load_dataset_by_key, list_dataset_cards
from fuzzyxai.pipelines import DatasetObserverPipeline, write_dataset_observer_report

ROOT = Path(__file__).resolve().parents[1]


def _read_json(path: str | Path) -> dict[str, Any]:
    """Backward-compatible helper used by smoke tests."""
    p = Path(path)
    if not p.exists():
        return {}
    return json.loads(p.read_text(encoding='utf-8'))


def _rows(df: pd.DataFrame, limit: int | None = None) -> list[dict[str, Any]]:
    if df.empty:
        return []
    out = df if limit is None else df.head(limit)
    return out.replace({np.nan: None}).to_dict(orient='records')


def _columns(df: pd.DataFrame) -> list[dict[str, str]]:
    return [{'name': c, 'label': c, 'field': c} for c in df.columns]


def _num(x: Any, nd: int = 4) -> str:
    try:
        return f'{float(x):.{nd}f}'
    except Exception:
        return '-'


def _chart_option(df: pd.DataFrame, x: str, y: str, title: str) -> dict[str, Any]:
    if df.empty or x not in df.columns or y not in df.columns:
        return {'title': {'text': f'{title} (missing data)'}}
    return {
        'title': {'text': title, 'left': 'center'},
        'tooltip': {'trigger': 'axis'},
        'xAxis': {'type': 'category', 'data': [str(v) for v in df[x].tolist()]},
        'yAxis': {'type': 'value'},
        'series': [{'type': 'line', 'smooth': True, 'data': [float(v) for v in df[y].tolist()]}],
    }


def _bar_option(df: pd.DataFrame, x: str, y: str, title: str) -> dict[str, Any]:
    if df.empty or x not in df.columns or y not in df.columns:
        return {'title': {'text': f'{title} (missing data)'}}
    return {
        'title': {'text': title, 'left': 'center'},
        'tooltip': {'trigger': 'axis'},
        'xAxis': {'type': 'category', 'data': [str(v) for v in df[x].tolist()]},
        'yAxis': {'type': 'value', 'min': 0, 'max': 1},
        'series': [{'type': 'bar', 'data': [float(v) for v in df[y].tolist()]}],
    }


def _sanitize_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    clean: list[dict[str, Any]] = []
    for row in rows:
        clean_row: dict[str, Any] = {}
        for k, v in row.items():
            if isinstance(v, (list, dict, tuple, set)):
                clean_row[k] = json.dumps(v, ensure_ascii=False)
            else:
                clean_row[k] = v
        clean.append(clean_row)
    return clean


def _action_card(ui: Any, title: str, value: str, tone: str = 'neutral') -> None:
    tone_class = {
        'ok': 'bg-emerald-50 border-emerald-200 text-emerald-900',
        'warn': 'bg-amber-50 border-amber-200 text-amber-900',
        'bad': 'bg-rose-50 border-rose-200 text-rose-900',
        'neutral': 'bg-slate-50 border-slate-200 text-slate-900',
    }.get(tone, 'bg-slate-50 border-slate-200 text-slate-900')
    with ui.column().classes(f'rounded-xl border p-3 gap-1 {tone_class}'):
        ui.label(title).classes('text-xs uppercase tracking-wide opacity-80')
        ui.label(value).classes('text-lg font-bold')


def _export_last_case(case: dict[str, Any]) -> None:
    if not case:
        return
    out_dir = ROOT / 'reports/unified_full_demo/live_cases'
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / 'last_case.json'
    path.write_text(json.dumps(case, ensure_ascii=False, indent=2), encoding='utf-8')


def run_ui(port: int = 8091) -> None:  # pragma: no cover - interactive app
    from nicegui import ui

    backend = build_backend()
    reports: HubReports = load_reports(ROOT)
    proba_all = backend.model.predict_proba(backend.x_test)[:, 0]
    low_idx = int(np.argmin(proba_all))
    high_idx = int(np.argmax(proba_all))
    amb_idx = int(np.argmin(np.abs(proba_all - 0.5)))

    ui.add_head_html(
        """
        <style>
        body { background: radial-gradient(1200px 600px at 20% -10%, #dbeafe 0%, rgba(219,234,254,0) 40%), radial-gradient(1000px 500px at 100% 0%, #dcfce7 0%, rgba(220,252,231,0) 35%), #f8fbff; }
        .shell { max-width: 1680px; margin: 0 auto; }
        .hero { background: linear-gradient(135deg,#0f172a,#1d4ed8,#0ea5e9); color: white; border-radius: 18px; padding: 20px; box-shadow: 0 16px 36px rgba(15,23,42,.24); }
        .panel { background: white; border: 1px solid #dbe4ef; border-radius: 14px; padding: 14px; box-shadow: 0 10px 24px rgba(20, 40, 70, .08); }
        </style>
        """
    )

    with ui.column().classes('shell w-full gap-4 p-4'):
        with ui.row().classes('hero w-full items-center justify-between'):
            with ui.column().classes('gap-1'):
                ui.label('FuzzyXAI Defense Hub').classes('text-3xl font-bold')
                ui.label('Единый GUI: глава 2, глава 3/5, Risk Observer, Category/HoTT, full pipeline').classes('opacity-90')
            with ui.column().classes('items-end gap-1 text-xs opacity-90'):
                ui.label('single source of truth')
                ui.label(f'features: {len(backend.feature_names)} | test samples: {len(backend.x_test)}')

        with ui.tabs().classes('w-full') as tabs:
            t_live = ui.tab('Live Case')
            t_risk = ui.tab('Observer & Ch2')
            t_exp = ui.tab('Experiments')
            t_cat = ui.tab('Category/HoTT')
            t_pred = ui.tab('Predictions')
            t_art = ui.tab('Artifacts')

        with ui.tab_panels(tabs, value=t_live).classes('w-full'):
            with ui.tab_panel(t_live):
                with ui.column().classes('panel w-full gap-3'):
                    ui.label('Dataset Pipeline Runner').classes('text-lg font-semibold')
                    cards = list_dataset_cards()
                    ds_options = {c.key: f'{c.name} [{c.domain}]' for c in cards}
                    ds_map = {c.key: c for c in cards}
                    ds_select = ui.select(ds_options, value='breast_cancer', label='Dataset').classes('w-full')
                    ds_meta = ui.markdown('')
                    ds_result = ui.markdown('')
                    ds_meta_table = ui.table(
                        columns=[
                            {'name': 'field', 'label': 'field', 'field': 'field'},
                            {'name': 'value', 'label': 'value', 'field': 'value'},
                        ],
                        rows=[],
                        pagination=20,
                    ).classes('w-full')
                    ds_run = ui.button('Run dataset benchmark', color='primary')

                    def _refresh_ds_meta() -> None:
                        key = str(ds_select.value)
                        card = ds_map[key]
                        ds_meta.content = (
                            f"**{card.name}**  \n"
                            f"source: `{card.source}`  \n"
                            f"license: `{card.license}`  \n"
                            f"domain: `{card.domain}`  \n"
                            f"desc: {card.description}"
                        )

                    def _run_ds() -> None:
                        key = str(ds_select.value)
                        card, record, df = load_dataset_by_key(key, allow_fallback=True)
                        pipeline = DatasetObserverPipeline(model_name='random_forest', mode='audit')
                        result = pipeline.run(record, df, target_column=record.target_column, case_index=0)
                        out_dir = ROOT / 'reports' / 'dataset_benchmark' / key
                        out_dir.mkdir(parents=True, exist_ok=True)
                        paths = write_dataset_observer_report(result, out_dir)
                        meta = {
                            'name': card.name,
                            'source': card.source,
                            'license': card.license,
                            'domain': card.domain,
                            'shape': f'{df.shape[0]}x{df.shape[1]}',
                            'features': len(df.columns) - 1,
                            'target': record.target_column,
                            'missing_values': float(df.isna().mean().mean()),
                            'label_description': f'target={record.target_column}',
                            'fallback': bool(record.metadata.get('fallback', False)),
                        }
                        ds_meta_table.rows = [{'field': k, 'value': v} for k, v in meta.items()]
                        ds_meta_table.update()
                        ds_result.content = (
                            f"**accuracy:** `{_num(result.accuracy)}`  \n"
                            f"**roc_auc:** `{_num(result.roc_auc) if result.roc_auc is not None else 'n/a'}`  \n"
                            f"**I_pre:** `{_num(result.observer_result.get('pre_interpretability'))}`  \n"
                            f"**rho:** `{_num(result.observer_result.get('application_risk'))}`  \n"
                            f"**action:** `{result.observer_result.get('action')}`  \n"
                            f"saved: `{paths['json']}`, `{paths['markdown']}`, `{paths['html']}`"
                        )

                    ds_select.on_value_change(lambda _e: _refresh_ds_meta())
                    ds_run.on_click(_run_ds)
                    _refresh_ds_meta()

                last_case: dict[str, Any] = {}
                with ui.row().classes('w-full gap-4'):
                    with ui.column().classes('panel w-2/5 gap-3'):
                        ui.label('Input').classes('text-lg font-semibold')
                        sample_select = ui.select(
                            options={i: f'Sample #{i} (true={int(backend.y_test.iloc[i])})' for i in range(len(backend.x_test))},
                            value=0,
                            label='Breast cancer sample',
                        ).classes('w-full')
                        with ui.row().classes('w-full gap-2'):
                            ui.button('Preset: safe', on_click=lambda: setattr(sample_select, 'value', low_idx)).classes('bg-emerald-500 text-white')
                            ui.button('Preset: ambiguous', on_click=lambda: setattr(sample_select, 'value', amb_idx)).classes('bg-amber-500 text-white')
                            ui.button('Preset: risky', on_click=lambda: setattr(sample_select, 'value', high_idx)).classes('bg-rose-500 text-white')
                        manual_toggle = ui.switch('Manual feature input', value=False)
                        inputs: dict[str, Any] = {}
                        with ui.expansion('Manual feature editor', value=False).classes('w-full'):
                            with ui.grid(columns=2).classes('w-full gap-2'):
                                for name in backend.feature_names:
                                    inputs[name] = ui.number(
                                        name,
                                        value=float(backend.x_test.iloc[0][name]),
                                        format='%.6f',
                                    ).classes('w-full')
                        run_btn = ui.button('Run observer').classes('w-full')

                    with ui.column().classes('panel w-3/5 gap-3'):
                        ui.label('Decision').classes('text-lg font-semibold')
                        result_md = ui.markdown('')
                        with ui.row().classes('w-full gap-3'):
                            box_action = ui.column().classes('w-1/3')
                            box_rho = ui.column().classes('w-1/3')
                            box_rupture = ui.column().classes('w-1/3')
                        context_table = ui.table(
                            columns=[
                                {'name': 'object', 'label': 'Object', 'field': 'object'},
                                {'name': 'RiskContext', 'label': 'RiskContext', 'field': 'RiskContext'},
                                {'name': 'AutoAccept', 'label': 'AutoAccept', 'field': 'AutoAccept'},
                            ],
                            rows=[],
                            pagination=8,
                        ).classes('w-full')
                        breakdown_table = ui.table(
                            columns=[
                                {'name': 'component', 'label': 'component', 'field': 'component'},
                                {'name': 'value', 'label': 'value', 'field': 'value'},
                            ],
                            rows=[],
                            pagination=10,
                        ).classes('w-full')
                        ui.button('Export this case to JSON', color='primary').on_click(
                            lambda: _export_last_case(last_case)
                        )

                def evaluate() -> None:
                    if manual_toggle.value:
                        vec = np.array([float(inputs[name].value) for name in backend.feature_names], dtype=float)
                        sample_id = 'manual'
                        true_y = '-'
                    else:
                        idx = int(sample_select.value)
                        vec = backend.x_test.iloc[idx].to_numpy(dtype=float)
                        sample_id = f'sample_{idx}'
                        true_y = int(backend.y_test.iloc[idx])

                    out = evaluate_vector(backend, vec, sample_id=sample_id)
                    last_case.clear()
                    last_case.update(out)
                    last_case['sample_id'] = sample_id
                    last_case['timestamp_utc'] = datetime.now(timezone.utc).isoformat()
                    result_md.content = (
                        f"**Prediction:** `{out['prediction']}`  |  "
                        f"**True y:** `{true_y}`  |  "
                        f"**P(malignant):** `{out['prob_malignant']:.4f}`  |  "
                        f"**I_pre:** `{out['I_pre']:.4f}`  |  "
                        f"**rho:** `{out['rho']:.4f}`  |  "
                        f"**Diagnostics:** `{','.join(out.get('diagnostics', [])) or '-'} `"
                    )

                    tone = 'ok'
                    if out['action'] in {'defer_to_human', 'request_more_data'}:
                        tone = 'warn'
                    if out['action'] == 'block':
                        tone = 'bad'

                    box_action.clear()
                    with box_action:
                        _action_card(ui, 'Action', str(out['action']), tone=tone)
                    box_rho.clear()
                    with box_rho:
                        _action_card(ui, 'Risk rho', _num(out['rho']), tone='warn' if out['rho'] >= 0.35 else 'ok')
                    box_rupture.clear()
                    with box_rupture:
                        _action_card(ui, 'Rupture', 'YES' if out['rupture'] else 'NO', tone='bad' if out['rupture'] else 'ok')

                    context_table.rows = out['contexts']
                    context_table.update()

                    rb = out.get('risk_breakdown', {})
                    breakdown_rows = [
                        {'component': 'predicted_risk', 'value': _num(rb.get('predicted_risk'))},
                        {'component': 'uncertainty', 'value': _num(rb.get('uncertainty'))},
                        {'component': 'interpretability_gap', 'value': _num(rb.get('interpretability_gap'))},
                        {'component': 'reduction_loss', 'value': _num(rb.get('reduction_loss'))},
                        {'component': 'diagnostic', 'value': _num(rb.get('diagnostic'))},
                    ]
                    breakdown_table.rows = breakdown_rows
                    breakdown_table.update()

                run_btn.on_click(evaluate)
                evaluate()

            with ui.tab_panel(t_risk):
                with ui.row().classes('w-full gap-3'):
                    with ui.column().classes('panel w-1/3 gap-1'):
                        ui.label('Chapter 2: Real Data').classes('text-lg font-semibold')
                        c2 = reports.chapter2_summary
                        ui.label(f"Model accuracy: {_num(c2.get('model_accuracy'))}")
                        ui.label(f"ROC AUC: {_num(c2.get('model_roc_auc'))}")
                        ui.label(f"I_pre mean: {_num(c2.get('i_pre_mean'))}")
                        ui.label(f"I_pre q05..q95: {_num(c2.get('i_pre_q05'))} .. {_num(c2.get('i_pre_q95'))}")
                    with ui.column().classes('panel w-1/3 gap-1'):
                        ui.label('Observer config').classes('text-lg font-semibold')
                        ui.label(f'weights: {backend.observer.weights}')
                        ui.label(f'thresholds: {backend.observer.thresholds}')
                        ui.label(f'gamma_max(plan): {backend.plan.beta.get("gamma_max")}')
                    with ui.column().classes('panel w-1/3 gap-1'):
                        fp = reports.full_pipeline_summary
                        ui.label('Full pipeline summary').classes('text-lg font-semibold')
                        ui.label(f"n evaluated: {fp.get('n_evaluated', '-')}")
                        ui.label(f"mean rho: {_num(fp.get('mean_rho'))}")
                        ui.label(f"rupture rate: {_num(fp.get('rupture_rate'))}")
                        ui.label(f"actions: {fp.get('actions', {})}")

            with ui.tab_panel(t_exp):
                with ui.row().classes('w-full gap-4'):
                    with ui.column().classes('panel w-1/2 gap-2'):
                        ui.label('Sensitivity: accuracy vs w_R').classes('text-lg font-semibold')
                        ui.echart(_chart_option(reports.sensitivity_wr, 'w_R', 'accuracy', 'Accuracy(w_R)')).classes('w-full h-72')
                    with ui.column().classes('panel w-1/2 gap-2'):
                        ui.label('Sensitivity: accuracy vs theta_high').classes('text-lg font-semibold')
                        ui.echart(_chart_option(reports.sensitivity_theta, 'theta_high', 'accuracy', 'Accuracy(theta_high)')).classes('w-full h-72')

                with ui.row().classes('w-full gap-4'):
                    with ui.column().classes('panel w-full gap-2'):
                        ui.label('Mode comparison (accuracy)').classes('text-lg font-semibold')
                        if reports.baseline.empty:
                            ui.label('missing reports/chapter5/baseline_comparison.csv')
                        else:
                            ui.echart(_bar_option(reports.baseline, 'mode', 'accuracy', 'Accuracy by mode')).classes('w-full h-72')

                with ui.row().classes('w-full gap-4'):
                    with ui.column().classes('panel w-2/3 gap-2'):
                        ui.label('Baseline comparison').classes('text-lg font-semibold')
                        if reports.baseline.empty:
                            ui.label('missing reports/chapter5/baseline_comparison.csv')
                        else:
                            ui.table(columns=_columns(reports.baseline), rows=_rows(reports.baseline), pagination=10).classes('w-full')
                    with ui.column().classes('panel w-1/3 gap-2'):
                        ui.label('Timing complexity').classes('text-lg font-semibold')
                        if reports.timing.empty:
                            ui.label('missing reports/chapter5/timing_complexity.csv')
                        else:
                            ui.table(columns=_columns(reports.timing), rows=_rows(reports.timing), pagination=8).classes('w-full')

            with ui.tab_panel(t_cat):
                cat = reports.category_hott_report
                with ui.row().classes('w-full gap-4'):
                    with ui.column().classes('panel w-1/3 gap-1'):
                        ui.label('Category/HoTT checks').classes('text-lg font-semibold')
                        ui.label(f"status: {cat.get('status', 'missing')}")
                        ui.label(f"checks: {len(cat.get('checks', []))}")
                        ui.label(f"objects: {len(cat.get('objects', []))}")
                        ui.label(f"morphisms: {len(cat.get('morphisms', []))}")
                        ui.label(f"paths: {len(cat.get('paths', []))}")
                        ui.label(f"ruptures: {len(cat.get('ruptures', []))}")
                    with ui.column().classes('panel w-2/3 gap-2'):
                        ui.label('Paths').classes('text-lg font-semibold')
                        path_rows = cat.get('paths', []) if isinstance(cat.get('paths', []), list) else []
                        if path_rows:
                            path_rows = _sanitize_rows(path_rows)
                            key_set: list[str] = sorted({k for r in path_rows for k in r.keys()})
                            ui.table(
                                columns=[{'name': k, 'label': k, 'field': k} for k in key_set],
                                rows=path_rows,
                                pagination=8,
                            ).classes('w-full')
                        else:
                            ui.label('No path rows in report (or report missing).')
                with ui.column().classes('panel w-full gap-2'):
                    ui.label('Presheaf contexts snapshot').classes('text-lg font-semibold')
                    ctx = cat.get('presheaf_contexts', {})
                    ctx_rows = [{'context': k, 'values': str(v)} for k, v in ctx.items()] if isinstance(ctx, dict) else []
                    if ctx_rows:
                        ui.table(
                            columns=[
                                {'name': 'context', 'label': 'context', 'field': 'context'},
                                {'name': 'values', 'label': 'values', 'field': 'values'},
                            ],
                            rows=ctx_rows,
                            pagination=8,
                        ).classes('w-full')
                    else:
                        ui.label('No presheaf_contexts in report.')

            with ui.tab_panel(t_pred):
                with ui.column().classes('panel w-full gap-2'):
                    ui.label('Full pipeline predictions (preview)').classes('text-lg font-semibold')
                    if reports.full_predictions.empty:
                        ui.label('missing reports/full_pipeline/predictions.csv')
                    else:
                        action_options = ['all'] + sorted(reports.full_predictions['action'].dropna().astype(str).unique().tolist())
                        action_filter = ui.select(action_options, value='all', label='Filter by action').classes('w-64')
                        pred_table = ui.table(
                            columns=_columns(reports.full_predictions),
                            rows=_rows(reports.full_predictions, limit=300),
                            pagination=20,
                        ).classes('w-full')

                        def apply_filter() -> None:
                            if action_filter.value == 'all':
                                df = reports.full_predictions
                            else:
                                df = reports.full_predictions[reports.full_predictions['action'].astype(str) == str(action_filter.value)]
                            pred_table.rows = _rows(df, limit=300)
                            pred_table.update()

                        action_filter.on_value_change(lambda _e: apply_filter())

            with ui.tab_panel(t_art):
                with ui.column().classes('panel w-full gap-2'):
                    ui.label('Artifacts and launch commands').classes('text-lg font-semibold')
                    ui.label('Legacy GUI entrypoints still available, but this hub is the main unified interface:')
                    ui.code('make unified-demo PORT=8091\nmake web-demo PORT=8090\nmake demo PORT=8085')
                    for rel in [
                        'reports/chapter2/chapter2_breast_cancer_summary.md',
                        'reports/chapter5/chapter5_experiments.md',
                        'reports/chapter5/chapter5_demo.json',
                        'reports/chapter5/baseline_comparison.csv',
                        'reports/chapter5/breast_cancer_validation.csv',
                        'reports/full_pipeline/summary.md',
                        'reports/full_pipeline/predictions.csv',
                        'reports/category_hott/category_hott_checks.md',
                        'docs/CATEGORICAL_HOTT_EXTENSION_RU.md',
                    ]:
                        p = ROOT / rel
                        if p.exists():
                            ui.link(rel, rel, new_tab=False)
                        else:
                            ui.label(f'missing: {rel}')

    ui.run(port=port, title='FuzzyXAI Defense Hub')


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--port', type=int, default=8091)
    args = parser.parse_args()
    run_ui(port=args.port)


if __name__ in {'__main__', '__mp_main__'}:
    main()
