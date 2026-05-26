"""Layered defense demo: one case through all dissertation layers."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

from fuzzyxai.datasets import list_dataset_modes, load_dataset_mode
from fuzzyxai.pipelines import DatasetObserverPipeline

from apps.services.hub_data import load_reports
from apps.services.layered_case import LayeredCaseService, build_case_state, export_case_state

ROOT = Path(__file__).resolve().parents[1]


def _num(x: Any, nd: int = 4) -> str:
    try:
        return f'{float(x):.{nd}f}'
    except Exception:
        return '-'


def _status_from_value(value: float, low: float, high: float) -> tuple[str, str]:
    if value < low:
        return 'низкий', '#16a34a'
    if value < high:
        return 'средний', '#d97706'
    return 'высокий', '#dc2626'


def _i_pre_comment(i_pre: float) -> tuple[str, str]:
    if i_pre >= 0.8:
        return 'объяснение устойчивое, можно доверять структуре', '#16a34a'
    if i_pre >= 0.6:
        return 'объяснение приемлемое, но есть потери интерпретируемости', '#d97706'
    return 'объяснение слабое, желательно ручная проверка', '#dc2626'


def _rho_comment(rho: float, thresholds: tuple[float, float, float, float]) -> tuple[str, str]:
    t1, t2, t3, t4 = thresholds
    if rho < t1:
        return 'риск низкий: авто-применение допустимо', '#16a34a'
    if rho < t2:
        return 'риск умеренный: применить с пониженным доверием', '#65a30d'
    if rho < t3:
        return 'риск заметный: запросить дополнительные данные', '#d97706'
    if rho < t4:
        return 'риск высокий: передать человеку', '#ea580c'
    return 'риск критический: блокировать автоматическое действие', '#dc2626'


def _save_trace(case_state: dict[str, Any]) -> Path:
    out = ROOT / 'reports/layered_demo'
    out.mkdir(parents=True, exist_ok=True)
    path = out / 'last_case_trace.json'
    payload = dict(case_state)
    payload['exported_at_utc'] = datetime.now(timezone.utc).isoformat()
    export_case_state(payload, path)
    return path


def _route_html(route: dict[str, str]) -> str:
    order = ['Input', 'Model', 'Omega', 'Expl', 'Fuzzy', 'Topos', 'Observer', 'Action']
    items = []
    for key in order:
        val = str(route.get(key, '-'))
        tone = '#22c55e'
        if 'rupture' in val or 'block' in val:
            tone = '#ef4444'
        elif 'selected' in val or 'decided' in val or 'morphism' in val:
            tone = '#3b82f6'
        items.append(
            f"<span style='display:inline-block;padding:6px 10px;border-radius:999px;margin:4px;border:1px solid #dbe4ef;background:#fff;'>"
            f"<b>{key}</b>: <span style='color:{tone}'>{val}</span></span>"
        )
    return ''.join(items)


def _membership_option(prob_malignant: float) -> dict[str, Any]:
    xs = [i / 100 for i in range(101)]

    def low(x: float) -> float:
        return max(0.0, 1.0 - x * 1.4)

    def med(x: float) -> float:
        return max(0.0, 1.0 - abs(x - 0.5) * 4.0)

    def high(x: float) -> float:
        return max(0.0, x)

    return {
        'tooltip': {'trigger': 'axis'},
        'xAxis': {'type': 'value', 'min': 0, 'max': 1},
        'yAxis': {'type': 'value', 'min': 0, 'max': 1},
        'series': [
            {'type': 'line', 'name': 'low', 'data': [[x, low(x)] for x in xs], 'smooth': True},
            {'type': 'line', 'name': 'medium', 'data': [[x, med(x)] for x in xs], 'smooth': True},
            {'type': 'line', 'name': 'high', 'data': [[x, high(x)] for x in xs], 'smooth': True},
            {'type': 'scatter', 'name': 'sample', 'symbolSize': 12, 'data': [[prob_malignant, high(prob_malignant)]]},
        ],
    }


def _risk_rows(components: dict[str, Any]) -> list[dict[str, Any]]:
    w = components.get('weights', {})
    return [
        {'component': 'predicted_risk', 'value': _num(components.get('predicted_risk')), 'weight': _num(w.get('predicted_risk')), 'contrib': _num(float(components.get('predicted_risk', 0.0)) * float(w.get('predicted_risk', 0.0)))},
        {'component': 'uncertainty', 'value': _num(components.get('uncertainty')), 'weight': _num(w.get('uncertainty')), 'contrib': _num(float(components.get('uncertainty', 0.0)) * float(w.get('uncertainty', 0.0)))},
        {'component': 'interpretability_gap', 'value': _num(components.get('interpretability_gap')), 'weight': _num(w.get('interpretability_gap')), 'contrib': _num(float(components.get('interpretability_gap', 0.0)) * float(w.get('interpretability_gap', 0.0)))},
        {'component': 'reduction_loss', 'value': _num(components.get('reduction_loss')), 'weight': _num(w.get('reduction_loss')), 'contrib': _num(float(components.get('reduction_loss', 0.0)) * float(w.get('reduction_loss', 0.0)))},
        {'component': 'diagnostic', 'value': _num(components.get('diagnostic')), 'weight': _num(w.get('diagnostic')), 'contrib': _num(float(components.get('diagnostic', 0.0)) * float(w.get('diagnostic', 0.0)))},
    ]


def _policy_rows(rho: float, thresholds: tuple[float, float, float, float], chi_r: int, chi_r_crit: int) -> list[dict[str, str]]:
    t1, t2, t3, t4 = thresholds
    defer_t = max(0.80, t4)
    rules = [
        ('chi_R^crit(x) = 1', 'block', chi_r_crit == 1),
        (f'rho(x) >= {defer_t:.2f} and chi_R^crit(x) = 0', 'defer_to_human', chi_r_crit == 0 and rho >= defer_t),
        (f'rho(x) < {t1:.2f} and chi_R^crit(x) = 0', 'accept', chi_r_crit == 0 and rho < t1),
        (f'{t1:.2f} <= rho(x) < {t2:.2f} and chi_R^crit(x) = 0', 'lower_confidence', chi_r_crit == 0 and t1 <= rho < t2),
        (f'{t2:.2f} <= rho(x) < {t3:.2f} and chi_R^crit(x) = 0', 'request_more_data', chi_r_crit == 0 and t2 <= rho < t3),
        (f'chi_R(x)=1 and chi_R^crit(x)=0 and rho(x) < {defer_t:.2f}', 'request_more_data', chi_r == 1 and chi_r_crit == 0 and rho < defer_t),
        (f'{t3:.2f} <= rho(x) < {defer_t:.2f} and chi_R^crit(x) = 0', 'defer_to_human', chi_r_crit == 0 and t3 <= rho < defer_t),
    ]
    return [
        {
            'condition': cond,
            'action': action,
            'active': 'да' if active else 'нет',
        }
        for cond, action, active in rules
    ]


def _line_option(x_values: list[Any], y_values: list[float], title: str) -> dict[str, Any]:
    return {
        'title': {'text': title, 'left': 'center'},
        'tooltip': {'trigger': 'axis'},
        'xAxis': {'type': 'category', 'data': [str(x) for x in x_values]},
        'yAxis': {'type': 'value'},
        'series': [{'type': 'line', 'smooth': True, 'data': [float(y) for y in y_values]}],
    }


def _dataset_status_rows() -> list[dict[str, str]]:
    domain_alias = {
        'medical': 'medical',
        'tabular_text': 'registry text-tabular',
        'medical_audit': 'medical audit',
        'industrial_cv': 'industrial CV',
        'general': 'tabular',
        'controlled': 'diagnostic',
    }
    validates_alias = {
        'breast_cancer': 'risk observer baseline',
        'diabetes_binary': 'borderline uncertainty',
        'wine_risk': 'transferability',
        'synthetic_ruptures': 'Expl/HoTT rupture logic',
        'registry_programs': 'external registry mode',
        'registry_mosmed_doctor_analysis': 'doctor/model audit',
        'registry_steel_ir': 'industrial transferability',
    }
    rows: list[dict[str, str]] = []
    for spec in list_dataset_modes():
        status = 'READY'
        try:
            load_dataset_mode(spec.key)
        except FileNotFoundError:
            status = 'MISSING'
        except Exception:
            status = 'ERROR'
        rows.append(
            {
                'dataset_mode': spec.key,
                'domain': domain_alias.get(spec.domain, spec.domain),
                'validates': validates_alias.get(spec.key, spec.purpose),
                'status': status,
            }
        )
    return rows


def _explain_block(
    ui: Any,
    *,
    title: str,
    question: str,
    formula: str | None = None,
    interpretation: str | None = None,
    conclusion: str | None = None,
) -> None:
    with ui.column().classes('w-full rounded-lg border border-sky-200 bg-sky-50 p-3 gap-1'):
        ui.label(title).classes('font-semibold text-sky-900')
        ui.markdown(f'**Вопрос:** {question}')
        if formula:
            ui.code(formula)
        if interpretation:
            ui.markdown(f'**Как читать результат:** {interpretation}')
        if conclusion:
            ui.markdown(f'**Вывод:** {conclusion}')


def run_ui(port: int = 8096) -> None:  # pragma: no cover
    from nicegui import ui

    service = LayeredCaseService.create()
    reports = load_reports(ROOT)
    scenario_map = {
        'operator_accept': 'safe',
        'uncertainty_reduction': 'ambiguous',
        'interface_rupture': 'rupture',
        'context_block': 'context_block',
        'full_risk_block': 'risky',
        'manual_breast_cancer': 'manual',
    }

    manual_vec = service.backend.x_test.iloc[0].to_numpy(dtype=float)
    state = build_case_state(service, 'safe')

    ui.add_head_html(
        """
        <style>
        body { background: linear-gradient(160deg,#eef5ff,#f9fcff 38%,#eefbf3); }
        .shell { max-width: 1640px; margin: 0 auto; }
        .hero { background: linear-gradient(135deg,#0f172a,#1d4ed8,#0ea5e9); color:#fff; border-radius:18px; padding:20px; }
        .panel { background:#fff; border:1px solid #dbe4ef; border-radius:14px; padding:14px; box-shadow:0 10px 24px rgba(20,40,70,.08); }
        </style>
        """
    )

    with ui.column().classes('shell w-full p-4 gap-4'):
        with ui.row().classes('hero w-full items-center justify-between'):
            with ui.column().classes('gap-1'):
                ui.label('FuzzyXAI Layered Demo').classes('text-3xl font-bold')
                ui.label('Input → Model → Ω → Expl → Fuzzy → Topos → Observer → Action').classes('opacity-90')
            ui.label('defense mode').classes('text-xs opacity-90')

        with ui.tabs().classes('w-full') as tabs:
            t_case = ui.tab('Сквозной кейс')
            t_op = ui.tab('Системный оператор Ω')
            t_fuzzy = ui.tab('Иерархия неопределённости')
            t_cat = ui.tab('Категория / HoTT / топос')
            t_risk = ui.tab('Риск-наблюдатель')
            t_exp = ui.tab('Эксперименты')
            t_art = ui.tab('Артефакты')

        with ui.tab_panels(tabs, value=t_case).classes('w-full'):
            with ui.tab_panel(t_case):
                modes = list_dataset_modes()
                mode_options = {m.key: f'{m.title} [{m.domain}]' for m in modes}
                mode_help = {m.key: m.purpose for m in modes}

                with ui.row().classes('w-full gap-4'):
                    with ui.column().classes('panel w-2/5 gap-2'):
                        ui.label('Dataset mode').classes('text-lg font-semibold')
                        dataset_mode = ui.select(mode_options, value='breast_cancer', label='Источник данных').classes('w-full')
                        dataset_purpose = ui.markdown(mode_help.get('breast_cancer', ''))
                        dataset_status = ui.markdown('')
                        ui.separator()
                        ui.label('Сценарий').classes('text-lg font-semibold')
                        scenario = ui.select(
                            {
                                'operator_accept': 'Безопасный кейс',
                                'uncertainty_reduction': 'Неопределённость и редукция',
                                'interface_rupture': 'Разрыв интерфейса',
                                'context_block': 'Контекстный запрет',
                                'full_risk_block': 'Критический риск',
                                'manual_breast_cancer': 'Ручной ввод',
                            },
                            value='operator_accept',
                            label='Сценарий',
                        ).classes('w-full')
                        sample_idx = ui.number('Sample index', value=0, min=0, step=1).classes('w-full')
                        manual_switch = ui.switch('Manual feature vector', value=False)
                        manual_inputs: dict[str, Any] = {}
                        with ui.expansion('Manual features', value=False).classes('w-full'):
                            with ui.grid(columns=2).classes('w-full gap-2'):
                                for i, feat in enumerate(service.backend.feature_names):
                                    manual_inputs[feat] = ui.number(feat, value=float(manual_vec[i]), format='%.6f').classes('w-full')
                        run_btn = ui.button('Run full pipeline', color='primary').classes('w-full')
                        export_btn = ui.button('Export trace JSON').classes('w-full')

                    with ui.column().classes('panel w-3/5 gap-2'):
                        ui.label('Итог кейса').classes('text-lg font-semibold')
                        mode_preview = ui.markdown('')
                        _explain_block(
                            ui,
                            title='Dataset modes: проверка переносимости',
                            question='На каких предметных режимах проверяется один и тот же контур?',
                            interpretation='Встроенные режимы дают воспроизводимую проверку без внешних файлов; registry-режимы показывают переносимость после локальной загрузки.',
                            conclusion='MISSING для registry не ошибка метода: это индикатор, что локальный файл ещё не подключён.',
                        )
                        mode_matrix = ui.table(
                            columns=[
                                {'name': 'dataset_mode', 'label': 'Dataset mode', 'field': 'dataset_mode'},
                                {'name': 'domain', 'label': 'Domain', 'field': 'domain'},
                                {'name': 'validates', 'label': 'What validates', 'field': 'validates'},
                                {'name': 'status', 'label': 'Status', 'field': 'status'},
                            ],
                            rows=_dataset_status_rows(),
                            pagination=8,
                        ).classes('w-full')
                        mode_matrix.add_slot(
                            'body-cell-status',
                            """
                            <q-td :props="props">
                              <q-badge
                                :color="props.value === 'READY' ? 'positive' : (props.value === 'MISSING' ? 'warning' : 'negative')"
                                text-color="white"
                                :label="props.value"
                              />
                            </q-td>
                            """,
                        )
                        _explain_block(
                            ui,
                            title='Сквозной кейс: от прогноза к действию',
                            question='Как этот конкретный объект проходит через весь наблюдающий контур?',
                            formula='Input -> Model -> Omega -> Expl -> Fuzzy -> Topos -> Risk -> Action',
                            interpretation='Смотрим не только вероятность модели, но и целостность объяснительной цепочки.',
                            conclusion='Итоговое действие выбирается только после проверки всех слоёв.',
                        )
                        route_md = ui.html('')
                        summary = ui.markdown('')
                        status_md = ui.html('')
                        step1 = ui.markdown('')
                        step2 = ui.markdown('')
                        step3 = ui.table(
                            columns=[
                                {'name': 'transition', 'label': 'Переход', 'field': 'transition'},
                                {'name': 'gamma', 'label': 'γ', 'field': 'gamma'},
                                {'name': 'gamma_max', 'label': 'γ_max', 'field': 'gamma_max'},
                                {'name': 'status', 'label': 'Результат', 'field': 'status'},
                                {'name': 'hott', 'label': 'HoTT', 'field': 'hott'},
                                {'name': 'meaning', 'label': 'Что это значит', 'field': 'meaning'},
                            ],
                            rows=[],
                        ).classes('w-full')
                        step4 = ui.table(
                            columns=[
                                {'name': 'class', 'label': 'Класс', 'field': 'class'},
                                {'name': 'coverage', 'label': 'Покрытие', 'field': 'coverage'},
                                {'name': 'complexity', 'label': 'Сложность', 'field': 'complexity'},
                                {'name': 'reduction_loss', 'label': 'Потеря редукции', 'field': 'reduction_loss'},
                                {'name': 'status', 'label': 'Статус', 'field': 'status'},
                            ],
                            rows=[],
                        ).classes('w-full')
                        step5 = ui.table(
                            columns=[
                                {'name': 'object', 'label': 'Объект', 'field': 'object'},
                                {'name': 'risk_context', 'label': 'RiskContext', 'field': 'risk_context'},
                                {'name': 'auto_accept', 'label': 'AutoAccept', 'field': 'auto_accept'},
                                {'name': 'conclusion', 'label': 'Вывод', 'field': 'conclusion'},
                            ],
                            rows=[],
                        ).classes('w-full')
                        step6 = ui.table(
                            columns=[
                                {'name': 'component', 'label': 'Компонента', 'field': 'component'},
                                {'name': 'value', 'label': 'Значение', 'field': 'value'},
                                {'name': 'weight', 'label': 'Вес', 'field': 'weight'},
                                {'name': 'contrib', 'label': 'Вклад', 'field': 'contrib'},
                            ],
                            rows=[],
                        ).classes('w-full')
                        final_card = ui.markdown('')

                dataset_mode.on_value_change(lambda _e: setattr(dataset_purpose, 'content', mode_help.get(str(dataset_mode.value), '')))

                def refresh() -> None:
                    nonlocal state
                    mode_matrix.rows = _dataset_status_rows()
                    mode_matrix.update()
                    scenario_key = scenario.value
                    mapped = scenario_map.get(str(scenario_key), 'safe')
                    if manual_switch.value and mapped == 'manual':
                        vec = np.array([float(manual_inputs[name].value) for name in service.backend.feature_names], dtype=float)
                        state = build_case_state(service, mapped, manual_vector=vec, dataset_mode=str(dataset_mode.value))
                    else:
                        state = build_case_state(service, mapped, sample_index=int(sample_idx.value), dataset_mode=str(dataset_mode.value))

                    route_md.content = _route_html(state['route_header'])
                    m = state['model']
                    r = state['risk']
                    i_pre = float(state['explanation']['I_pre'])
                    i_pre_text, i_pre_color = _i_pre_comment(i_pre)
                    rho_text, rho_color = _rho_comment(float(r['rho']), tuple(r['thresholds']))

                    rupture_text = 'есть разрыв (нельзя слепо авто-применять)' if r['rupture'] else 'разрыва нет (цепочка согласована)'
                    rupture_color = '#dc2626' if r['rupture'] else '#16a34a'

                    summary.content = (
                        f"| Metric | Value |\n|---|---|\n"
                        f"| Prediction | `{m['prediction']}` |\n"
                        f"| P(malignant) | `{_num(m['p_malignant'])}` |\n"
                        f"| I_pre | `{_num(state['explanation']['I_pre'])}` |\n"
                        f"| rho | `{_num(r['rho'])}` |\n"
                        f"| Rupture | `{'yes' if r['rupture'] else 'no'}` |\n"
                        f"| Action | `{r['action']}` |"
                    )

                    status_md.content = (
                        f"<div style='display:flex;gap:10px;flex-wrap:wrap'>"
                        f"<div style='padding:8px 12px;border-radius:10px;background:#f8fafc;border:1px solid #dbe4ef'><b>I_pre</b>: <span style='color:{i_pre_color}'>{i_pre_text}</span></div>"
                        f"<div style='padding:8px 12px;border-radius:10px;background:#f8fafc;border:1px solid #dbe4ef'><b>Rupture</b>: <span style='color:{rupture_color}'>{rupture_text}</span></div>"
                        f"<div style='padding:8px 12px;border-radius:10px;background:#f8fafc;border:1px solid #dbe4ef'><b>rho</b>: <span style='color:{rho_color}'>{rho_text}</span></div>"
                        f"</div>"
                    )

                    step1.content = (
                        f"**Шаг 1. Вход и модель**  \n"
                        f"sample: `{state['input']['sample_id']}`; true_y=`{m['true_y']}`; "
                        f"pred=`{m['prediction']}`; predicted_risk=`{_num(m['predicted_risk'])}`"
                    )
                    e = state['explanation']['E_model']
                    step2.content = (
                        f"**Шаг 2. Ω-оператор**  \n"
                        f"`E_model=<L, μ, R, α, u, τ>`  \n"
                        f"L={e['L']} | α={e['alpha']} | u={_num(e['u'])} | τ={e['tau']}"
                    )

                    edge_rows = []
                    for row in state['composition']['edges']:
                        gamma_val = row['gamma']
                        if gamma_val is None:
                            meaning = 'нет численной оценки: переход диагностирован как разрыв'
                        else:
                            gv = float(gamma_val)
                            gmax = float(row['gamma_max'])
                            if gv <= gmax:
                                meaning = 'переход допустим: морфизм существует'
                            else:
                                meaning = 'переход недопустим: порог превышен, rupture'
                        edge_rows.append(
                            {
                                'transition': f"{row['source']} -> {row['target']}",
                                'gamma': _num(row['gamma']),
                                'gamma_max': _num(row['gamma_max']),
                                'status': row['status'],
                                'hott': row['hott'],
                                'meaning': meaning,
                            }
                        )
                    step3.rows = edge_rows
                    step3.update()

                    step4.rows = state['uncertainty']['classes']
                    step4.update()

                    ctx = state['contexts']
                    step5.rows = [
                        {
                            'object': 'E_model',
                            'risk_context': ctx['RiskContext']['E_model'],
                            'auto_accept': ctx['AutoAccept']['E_model'],
                            'conclusion': 'авто возможно' if ctx['AutoAccept']['E_model'] else 'авто запрещено',
                        },
                        {
                            'object': 'E_risk',
                            'risk_context': ctx['RiskContext']['E_risk'],
                            'auto_accept': ctx['AutoAccept']['E_risk'],
                            'conclusion': 'авто возможно' if ctx['AutoAccept']['E_risk'] else 'авто запрещено',
                        },
                        {
                            'object': 'E_action',
                            'risk_context': ctx['RiskContext']['E_action'],
                            'auto_accept': ctx['AutoAccept']['E_action'],
                            'conclusion': 'авто возможно' if ctx['AutoAccept']['E_action'] else 'авто запрещено',
                        },
                    ]
                    step5.update()

                    step6.rows = _risk_rows(r['components'])
                    step6.update()

                    action = str(r['action']).upper()
                    reason = r['reason']
                    final_card.content = (
                        f"### Action: **{action}**\n"
                        f"Причина: `{reason}`.  \n"
                        f"Что с того: **{rho_text}**.  \n"
                        f"Пороги: `{r['thresholds']}`"
                    )

                    mode_key = str(dataset_mode.value)
                    try:
                        record, df = load_dataset_mode(mode_key)
                        dataset_status.content = (
                            f"**Статус:** READY  \n"
                            f"mode=`{mode_key}`, rows=`{len(df)}`, target=`{record.target_column}`"
                        )
                        try:
                            pipeline = DatasetObserverPipeline(model_name='random_forest', mode='audit')
                            ds_result = pipeline.run(record, df, case_index=0)
                            mode_preview.content = (
                                f"**Dataset mode check:** `{mode_key}`  \n"
                                f"Domain: `{record.metadata.get('domain', 'n/a')}`  \n"
                                f"Action: `{ds_result.observer_result['action']}`, "
                                f"rho=`{_num(ds_result.observer_result['application_risk'])}`, "
                                f"repr=`{ds_result.observer_result['selected_representation']}`  \n"
                                f"Observer accuracy: `{_num(ds_result.observer_action_accuracy)}`"
                            )
                        except Exception as exc:
                            mode_preview.content = (
                                f"**Dataset mode check:** `{mode_key}`  \n"
                                f"Pipeline status: failed  \n"
                                f"`{type(exc).__name__}: {exc}`"
                            )
                    except FileNotFoundError as exc:
                        dataset_status.content = f"**Статус:** MISSING  \n`{exc}`"
                        mode_preview.content = (
                            f"**Dataset mode check:** `{mode_key}`  \n"
                            f"Локальный файл не найден. Для загрузки: см. `docs/REGISTRY_DATASETS_SETUP_RU.md`."
                        )
                    except Exception as exc:
                        dataset_status.content = f"**Статус:** ERROR  \n`{type(exc).__name__}: {exc}`"
                        mode_preview.content = f"**Dataset mode check:** `{mode_key}`  \nОшибка подготовки режима."

                run_btn.on_click(refresh)
                export_btn.on_click(lambda: _save_trace(state))
                dataset_mode.on_value_change(lambda _e: refresh())
                scenario.on_value_change(lambda _e: refresh())
                sample_idx.on_value_change(lambda _e: refresh())
                manual_switch.on_value_change(lambda _e: refresh())
                dataset_mode.on_value_change(lambda _e: refresh())
                scenario.on_value_change(lambda _e: refresh())
                sample_idx.on_value_change(lambda _e: refresh())
                manual_switch.on_value_change(lambda _e: refresh())
                refresh()

            with ui.tab_panel(t_op):
                with ui.column().classes('panel w-full gap-2'):
                    ui.label('Оператор Ω').classes('text-lg font-semibold')
                    _explain_block(
                        ui,
                        title='Системный оператор Ω (глава 2)',
                        question='Удалось ли построить проверяемый объяснительный объект, а не просто текст-комментарий?',
                        formula='E_model = <L, μ, R, α, u, τ>',
                        interpretation='Если объект построен, его можно согласовывать с другими модулями и аудировать.',
                        conclusion='Оператор переводит состояние модели в структуру, пригодную для контроля и композиции.',
                    )
                    chart = ui.echart(_membership_option(state['model']['p_malignant'])).classes('w-full h-80')
                    ui.markdown('**Что показывает график:** как числовой риск переводится в термы `low/medium/high`.')
                    details = ui.table(
                        columns=[
                            {'name': 'field', 'label': 'Поле', 'field': 'field'},
                            {'name': 'value', 'label': 'Оценка', 'field': 'value'},
                            {'name': 'meaning', 'label': 'Что означает', 'field': 'meaning'},
                        ],
                        rows=[],
                    ).classes('w-full')

                    def refresh_op() -> None:
                        e = state['explanation']['E_model']
                        chart.options.clear()
                        chart.options.update(_membership_option(state['model']['p_malignant']))
                        chart.update()
                        u_value = float(e['u'])
                        u_level, _ = _status_from_value(u_value, 0.2, 0.5)
                        i_pre_val = float(state['explanation']['I_pre'])
                        i_pre_text, _ = _i_pre_comment(i_pre_val)
                        details.rows = [
                            {'field': 'L', 'value': str(e['L']), 'meaning': 'словарь лингвистических термов'},
                            {'field': 'μ(low/high)', 'value': str(e['mu']), 'meaning': 'как риск переводится в термы'},
                            {'field': 'R', 'value': str(e['R']), 'meaning': 'какие правила активированы'},
                            {'field': 'α', 'value': str(e['alpha']), 'meaning': 'насколько сильно сработали правила'},
                            {'field': 'u', 'value': _num(u_value), 'meaning': f'неопределённость {u_level}: чем выше, тем осторожнее решение'},
                            {'field': 'I_pre', 'value': _num(i_pre_val), 'meaning': i_pre_text},
                            {'field': 'τ', 'value': str(e['tau']), 'meaning': 'трассировка для аудита и воспроизводимости'},
                        ]
                        details.update()

                    ui.button('Refresh from current case', on_click=refresh_op)
                    refresh_op()

            with ui.tab_panel(t_fuzzy):
                with ui.column().classes('panel w-full gap-2'):
                    ui.label('Иерархия неопределённости').classes('text-lg font-semibold')
                    _explain_block(
                        ui,
                        title='Выбор класса неопределённости (глава 3)',
                        question='Какой класс представления нужен, чтобы не потерять смысл неопределённости?',
                        formula='Select(F) by coverage + complexity + reduction loss Delta',
                        interpretation='Сначала проверяем покрытие типов неопределённости, потом выбираем минимально достаточный класс.',
                        conclusion='Выбирается не самый сложный класс, а минимально достаточный для текущего профиля.',
                    )
                    ui.markdown(
                        '**Что это:** как именно мы представляем неопределённость (простое число, интервал, конфликт экспертов и т.д.).  \n'
                        '**Зачем:** чтобы не потерять важную информацию при объяснении.  \n'
                        '**Идея:** выбираем минимально сложный класс, который покрывает нужные типы неопределённости.'
                    )
                    profile_md = ui.markdown('')
                    profile_table = ui.table(
                        columns=[
                            {'name': 'type', 'label': 'Тип неопределённости', 'field': 'type'},
                            {'name': 'active', 'label': 'Активен', 'field': 'active'},
                            {'name': 'meaning', 'label': 'Что это значит', 'field': 'meaning'},
                        ],
                        rows=[],
                    ).classes('w-full')
                    glossary_table = ui.table(
                        columns=[
                            {'name': 'class', 'label': 'Класс', 'field': 'class'},
                            {'name': 'plain', 'label': 'Простое объяснение', 'field': 'plain'},
                        ],
                        rows=[
                            {'class': 'F0', 'plain': 'обычная нечёткая оценка (одно число 0..1)'},
                            {'class': 'F_int', 'plain': 'интервал значений, когда точное число неизвестно'},
                            {'class': 'F_H', 'plain': 'несколько экспертных оценок одновременно'},
                            {'class': 'F_N_src', 'plain': 'есть поддержка/сомнение/опровержение из разных источников'},
                            {'class': 'FML-audit', 'plain': 'многоуровневый режим для строгого аудита'},
                        ],
                    ).classes('w-full')
                    table = ui.table(
                        columns=[
                            {'name': 'class', 'label': 'Класс', 'field': 'class'},
                            {'name': 'coverage', 'label': 'Покрытие', 'field': 'coverage'},
                            {'name': 'complexity', 'label': 'Сложность', 'field': 'complexity'},
                            {'name': 'reduction_loss', 'label': 'Потеря редукции', 'field': 'reduction_loss'},
                            {'name': 'status', 'label': 'Статус', 'field': 'status'},
                        ],
                        rows=[],
                    ).classes('w-full')
                    fuzzy_conclusion = ui.markdown('')

                    def refresh_fuzzy() -> None:
                        u = state['uncertainty']
                        profile = u['profile']
                        profile_table.rows = [
                            {'type': 'u_num', 'active': 'yes' if profile.get('u_num') else 'no', 'meaning': 'базовая численная неопределённость модели'},
                            {'type': 'u_int', 'active': 'yes' if profile.get('u_int') else 'no', 'meaning': 'значение задано интервалом, а не точкой'},
                            {'type': 'u_expert', 'active': 'yes' if profile.get('u_expert') else 'no', 'meaning': 'эксперты/источники могут не соглашаться'},
                            {'type': 'u_conf', 'active': 'yes' if profile.get('u_conf') else 'no', 'meaning': 'есть структурный конфликт/разрыв'},
                            {'type': 'u_trace', 'active': 'yes' if profile.get('u_trace') else 'no', 'meaning': 'важна полная прослеживаемость для аудита'},
                        ]
                        profile_table.update()

                        profile_md.content = (
                            f"Selected representation: **{u['selected_class']}**  \n"
                            f"Delta = `{_num(u['delta'])}`  \n"
                            f"Что с того: чем меньше Delta, тем меньше потерь при упрощении объяснения."
                        )
                        table.rows = u['classes']
                        table.update()
                        fuzzy_conclusion.content = (
                            f"**Вывод по кейсу:** выбран `{u['selected_class']}`. "
                            f"Это даёт покрытие нужных неопределённостей при потере `Delta={_num(u['delta'])}`."
                        )

                    ui.button('Refresh from current case', on_click=refresh_fuzzy)
                    refresh_fuzzy()

            with ui.tab_panel(t_cat):
                with ui.column().classes('panel w-full gap-2'):
                    ui.label('Категория / HoTT / топос').classes('text-lg font-semibold')
                    _explain_block(
                        ui,
                        title='Проверка согласования объяснений',
                        question='Можно ли корректно передать смысл между слоями без искажения?',
                        formula='E_model -> E_risk -> E_action',
                        interpretation='Morphism = переход допустим; Rupture = переход недопустим и требует безопасного режима.',
                        conclusion='Category/HoTT слой даёт структурный тест: путь существует или есть разрыв.',
                    )
                    ui.markdown(
                        '**Очень просто:** мы проверяем, можно ли без искажения передать смысл объяснения между шагами.  \n'
                        '- Если можно: это *morphism* (зелёный переход).  \n'
                        '- Если нельзя: это *rupture* (красный разрыв).  \n'
                        '**Topos-часть:** смотрим, какие действия допустимы в каждом контексте (RiskContext) и где авто-режим запрещён (AutoAccept пусто).'
                    )
                    cat_table = ui.table(
                        columns=[
                            {'name': 'transition', 'label': 'Transition', 'field': 'transition'},
                            {'name': 'status', 'label': 'Status', 'field': 'status'},
                            {'name': 'hott', 'label': 'HoTT', 'field': 'hott'},
                            {'name': 'meaning', 'label': 'Что это значит', 'field': 'meaning'},
                        ],
                        rows=[],
                    ).classes('w-full')
                    ctx_table = ui.table(
                        columns=[
                            {'name': 'object', 'label': 'Object', 'field': 'object'},
                            {'name': 'RiskContext', 'label': 'RiskContext', 'field': 'RiskContext'},
                            {'name': 'AutoAccept', 'label': 'AutoAccept', 'field': 'AutoAccept'},
                            {'name': 'decision_hint', 'label': 'Практический вывод', 'field': 'decision_hint'},
                        ],
                        rows=[],
                    ).classes('w-full')
                    cat_conclusion = ui.markdown('')
                    _explain_block(
                        ui,
                        title='Topos-контекст действий',
                        question='Где действие допустимо автоматически, а где запрещено контекстом?',
                        formula='AutoAccept(E) subset RiskContext(E)',
                        interpretation='Если AutoAccept пусто, авто-применение запрещено даже при умеренном риске.',
                        conclusion='Контекстная логика предотвращает опасные авто-решения.',
                    )

                    def refresh_cat() -> None:
                        rows = []
                        for edge in state['composition']['edges']:
                            meaning = 'смысл передан корректно' if edge['status'] == 'morphism' else 'смысл не согласован, нужен безопасный режим'
                            rows.append({'transition': f"{edge['source']} -> {edge['target']}", 'status': edge['status'], 'hott': edge['hott'], 'meaning': meaning})
                        cat_table.rows = rows
                        cat_table.update()
                        ctx = state['contexts']
                        ctx_table.rows = [
                            {
                                'object': 'E_model',
                                'RiskContext': ctx['RiskContext']['E_model'],
                                'AutoAccept': ctx['AutoAccept']['E_model'],
                                'decision_hint': 'авто-режим возможен' if ctx['AutoAccept']['E_model'] else 'авто-режим запрещён',
                            },
                            {
                                'object': 'E_risk',
                                'RiskContext': ctx['RiskContext']['E_risk'],
                                'AutoAccept': ctx['AutoAccept']['E_risk'],
                                'decision_hint': 'авто-режим возможен' if ctx['AutoAccept']['E_risk'] else 'авто-режим запрещён',
                            },
                            {
                                'object': 'E_action',
                                'RiskContext': ctx['RiskContext']['E_action'],
                                'AutoAccept': ctx['AutoAccept']['E_action'],
                                'decision_hint': 'авто-режим возможен' if ctx['AutoAccept']['E_action'] else 'авто-режим запрещён',
                            },
                        ]
                        ctx_table.update()

                        has_rupture = any(edge['status'] != 'morphism' for edge in state['composition']['edges'])
                        if has_rupture:
                            cat_conclusion.content = '### Вывод: есть разрыв между слоями, поэтому контур обязан включить безопасное действие (defer/block).'
                        else:
                            cat_conclusion.content = '### Вывод: переходы согласованы, контур может выбирать мягкие действия по уровню риска.'

                    ui.button('Refresh from current case', on_click=refresh_cat)
                    refresh_cat()

            with ui.tab_panel(t_risk):
                with ui.column().classes('panel w-full gap-2'):
                    ui.label('Риск-наблюдатель').classes('text-lg font-semibold')
                    _explain_block(
                        ui,
                        title='Решение наблюдателя',
                        question='Какое действие безопасно с учётом прогноза и целостности объяснения?',
                        formula='rho = w_p*rho_p + w_u*u + w_I*(1-I_pre) + w_Delta*Delta + w_R*chi_R',
                        interpretation='Считается вклад каждой компоненты; затем применяется пороговая политика и правило критического разрыва.',
                        conclusion='Действие выбирается по интегральному риску, а не по одной вероятности модели.',
                    )
                    table = ui.table(
                        columns=[
                            {'name': 'component', 'label': 'Компонента', 'field': 'component'},
                            {'name': 'value', 'label': 'Значение', 'field': 'value'},
                            {'name': 'weight', 'label': 'Вес', 'field': 'weight'},
                            {'name': 'contrib', 'label': 'Вклад', 'field': 'contrib'},
                        ],
                        rows=[],
                    ).classes('w-full')
                    policy_table = ui.table(
                        columns=[
                            {'name': 'condition', 'label': 'Условие', 'field': 'condition'},
                            {'name': 'action', 'label': 'Действие', 'field': 'action'},
                            {'name': 'active', 'label': 'Активно', 'field': 'active'},
                        ],
                        rows=[],
                    ).classes('w-full')
                    policy_md = ui.markdown('')

                    def refresh_risk() -> None:
                        table.rows = _risk_rows(state['risk']['components'])
                        table.update()
                        thresholds = state['risk']['thresholds']
                        rho_val = float(state['risk']['rho'])
                        rupture = bool(state['risk']['rupture'])
                        policy_table.rows = _policy_rows(
                            rho_val,
                            tuple(thresholds),
                            int(state['risk'].get('chi_R', 0)),
                            int(state['risk'].get('chi_R_crit', 0)),
                        )
                        policy_table.update()
                        rho_text, _ = _rho_comment(float(state['risk']['rho']), tuple(thresholds))
                        policy_md.content = (
                            f"Thresholds: `{thresholds}`  \n"
                            f"Action: **{state['risk']['action']}**; chi_R=`{state['risk'].get('chi_R', 0)}`; "
                            f"chi_R^crit=`{state['risk'].get('chi_R_crit', 0)}`  \n"
                            f"Вывод: {rho_text}"
                        )

                    ui.button('Refresh from current case', on_click=refresh_risk)
                    refresh_risk()

            with ui.tab_panel(t_exp):
                with ui.column().classes('panel w-full gap-2'):
                    ui.label('Эксперименты').classes('text-lg font-semibold')
                    _explain_block(
                        ui,
                        title='Сравнение режимов',
                        question='Даёт ли полный контур преимущество над упрощёнными baseline?',
                        interpretation='Сравниваем accuracy, false blocks и missed ruptures.',
                        conclusion='Ключевая цель: не пропускать критические разрывы даже ценой более осторожной политики.',
                    )
                    if reports.baseline.empty:
                        ui.label('baseline report missing')
                    else:
                        ui.table(
                            columns=[{'name': c, 'label': c, 'field': c} for c in reports.baseline.columns],
                            rows=reports.baseline.to_dict(orient='records'),
                            pagination=10,
                        ).classes('w-full')

                    _explain_block(
                        ui,
                        title='Чувствительность к theta_high',
                        question='Насколько результат зависит от порога высокого риска?',
                        interpretation='Если accuracy заметно меняется, порог нужно калибровать на валидации.',
                        conclusion='theta_high — управленческий параметр: его нельзя фиксировать без калибровки.',
                    )
                    if reports.sensitivity_theta.empty:
                        ui.label('sensitivity_theta_high missing')
                    else:
                        ui.echart(
                            _line_option(
                                reports.sensitivity_theta['theta_high'].tolist(),
                                reports.sensitivity_theta['accuracy'].tolist(),
                                'Accuracy vs theta_high',
                            )
                        ).classes('w-full h-64')

                    _explain_block(
                        ui,
                        title='Чувствительность к w_R',
                        question='Насколько поведение наблюдателя зависит от веса диагностического разрыва?',
                        formula='rho = ... + w_R * chi_R',
                        interpretation='Смотрим как меняется accuracy при варьировании w_R.',
                        conclusion='Безопасность определяется не только весом, но и структурным правилом для критических разрывов.',
                    )
                    if reports.sensitivity_wr.empty:
                        ui.label('sensitivity_w_R missing')
                    else:
                        ui.echart(
                            _line_option(
                                reports.sensitivity_wr['w_R'].tolist(),
                                reports.sensitivity_wr['accuracy'].tolist(),
                                'Accuracy vs w_R',
                            )
                        ).classes('w-full h-64')

                    if reports.sensitivity_wr.empty:
                        ui.label('block_rate for w_R unavailable')
                    else:
                        _explain_block(
                            ui,
                            title='Стабильность блокировок',
                            question='Меняется ли доля блокировок при изменении w_R?',
                            interpretation='Смотрим block_rate(w_R): если сильно не плывёт, работает структурный safeguard.',
                            conclusion='Критические разрывы обрабатываются правилом запрета, а не только подбором веса.',
                        )
                        ui.echart(
                            _line_option(
                                reports.sensitivity_wr['w_R'].tolist(),
                                reports.sensitivity_wr['block_rate'].tolist(),
                                'Block rate vs w_R',
                            )
                        ).classes('w-full h-64')

            with ui.tab_panel(t_art):
                with ui.column().classes('panel w-full gap-2'):
                    ui.label('Артефакты и воспроизводимость').classes('text-lg font-semibold')
                    ui.code(
                        'make layered-demo PORT=8096\n'
                        'make unified-demo PORT=8095\n'
                        'make unified-demo-cli\n'
                        'make dataset-modes-check'
                    )
                    for rel in [
                        'reports/chapter5/breast_cancer_validation.csv',
                        'reports/chapter5/chapter5_experiments.json',
                        'reports/full_pipeline/summary.json',
                        'reports/unified_full_demo/unified_full_demo_report.json',
                        'reports/layered_demo/last_case_trace.json',
                    ]:
                        p = ROOT / rel
                        if p.exists():
                            ui.link(rel, rel, new_tab=False)
                        else:
                            ui.label(f'missing: {rel}')

    ui.run(port=port, title='FuzzyXAI Layered Demo')


def _route_statuses(state: dict[str, Any]) -> list[str]:
    return [str(v) for v in state.get('route_header', {}).values()]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--port', type=int, default=8096)
    args = parser.parse_args()
    run_ui(port=args.port)


if __name__ in {'__main__', '__mp_main__'}:
    main()
