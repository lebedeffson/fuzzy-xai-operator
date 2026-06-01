from __future__ import annotations

import argparse
import csv
import hashlib
import json
import platform
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Any

from fuzzyxai.studio import StudioExplainPlan


ROOT = Path(__file__).resolve().parents[1]


def _canonical_json(data: Any) -> str:
    return json.dumps(data, ensure_ascii=False, sort_keys=True, separators=(',', ':'))


def _sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode('utf-8')).hexdigest()


def _sha256_file(path: Path) -> str | None:
    if not path.exists() or not path.is_file():
        return None
    h = hashlib.sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def _artifact(path: str, confirms: str, chapter: str, command: str = '') -> dict[str, Any]:
    p = ROOT / path
    return {
        'path': path,
        'exists': p.exists(),
        'sha256': _sha256_file(p),
        'chapter': chapter,
        'confirms': confirms,
        'command': command,
    }


def _load_json_rel(path: str) -> dict[str, Any]:
    p = ROOT / path
    if not p.exists():
        return {}
    return json.loads(p.read_text(encoding='utf-8'))


def _fmt(value: Any, nd: int = 4) -> str:
    if value is None:
        return 'N/A'
    if isinstance(value, bool):
        return 'true' if value else 'false'
    if isinstance(value, (int, float)):
        return f'{float(value):.{nd}f}'
    return str(value)


def _row(cells: list[Any]) -> str:
    return '| ' + ' | '.join(str(c) for c in cells) + ' |'


def build_payload() -> dict[str, Any]:
    plan = StudioExplainPlan()
    plan_payload = asdict(plan)
    plan_canonical = _canonical_json(plan_payload)
    plan_hash = _sha256_text(plan_canonical)

    artifacts = [
        _artifact(
            'reports/reproducibility_artifacts/explain_plan.json',
            'Serializable ExplainPlan contract and trace hash source',
            '2 / appendix',
            'make reproducibility-artifacts',
        ),
        _artifact(
            'reports/chapter2_real_operator_case/breast_cancer_operator_case.json',
            'sample_113 operator values: mu, alpha, U_model, U_rules, U_trace, u_M, tau',
            '2',
            'make chapter2-real-operator-case',
        ),
        _artifact(
            'reports/real_reduction_example/breast_cancer_case.json',
            'sample_113 reduction, chi_Auto, chi_R, chi_R_crit, rho, action',
            '3',
            'make real-reduction-example',
        ),
        _artifact(
            'reports/datasets/breast_cancer/summary.json',
            'Breast Cancer model metrics, observer metrics, I_pre/rho quantiles',
            '5',
            'make benchmark-dataset DATASET=breast_cancer',
        ),
        _artifact(
            'reports/datasets/breast_cancer/calibration.json',
            'Observer calibration before/after and constrained parameters',
            '5',
            'make calibrate-observer DATASET=breast_cancer',
        ),
        _artifact(
            'reports/datasets/synthetic_ruptures/baseline_comparison_native.json',
            'Native-access safety comparison; baselines do not receive chi_R_crit',
            '5',
            'make baseline-comparison DATASET=synthetic_ruptures BASELINE_ACCESS=native',
        ),
        _artifact(
            'reports/datasets/synthetic_ruptures/baseline_comparison_equal_guardrail.json',
            'Equal-guardrail sanity check where all methods receive chi_R_crit',
            '5',
            'make baseline-comparison DATASET=synthetic_ruptures BASELINE_ACCESS=equal_guardrail',
        ),
        _artifact(
            'reports/structure_aware_benchmark/breast_cancer.json',
            'Structure-aware cases: trace gap, source conflict, context forbidden, high Delta',
            '5',
            'make structure-aware-benchmark DATASET=breast_cancer',
        ),
        _artifact(
            'reports/structure_aware_benchmark/wine_risk.json',
            'Structure-aware benchmark on real wine rows (non-synthetic improvement check)',
            '5',
            'make structure-aware-benchmark DATASET=wine_risk',
        ),
        _artifact(
            'reports/structure_aware_benchmark/diabetes_binary.json',
            'Structure-aware benchmark on real diabetes rows (non-synthetic improvement check)',
            '5',
            'make structure-aware-benchmark DATASET=diabetes_binary',
        ),
        _artifact(
            'reports/datasets/breast_cancer/ablation.json',
            'Ablation of trace, Delta, chi_R_crit, hierarchy, topos, risk-only threshold',
            '5',
            'make ablation-benchmark DATASET=breast_cancer',
        ),
        _artifact(
            'reports/thesis_practice/thesis_practice_tables.json',
            'Word/LaTeX-ready thesis practice table index',
            'appendix',
            'make thesis-practice-tables',
        ),
        _artifact(
            'Makefile',
            'Reproducible command route',
            'appendix',
            '',
        ),
        _artifact(
            'requirements.txt',
            'Python environment dependencies',
            'appendix',
            '',
        ),
    ]

    return {
        'status': 'ok',
        'environment': {
            'python': sys.version.split()[0],
            'platform': platform.platform(),
        },
        'explain_plan': {
            'canonical_sha256': plan_hash,
            'canonical_json': plan_payload,
        },
        'commands': [
            'make dissertation-check',
            'make thesis-practice-tables',
        'make browser-visual-check',
        'make ui-health-check',
        'make structure-aware-benchmark DATASET=wine_risk',
        'make structure-aware-benchmark DATASET=diabetes_binary',
        'make reproducibility-artifacts',
    ],
        'artifacts': artifacts,
    }


def build_article_insert(payload: dict[str, Any]) -> str:
    operator = _load_json_rel('reports/chapter2_real_operator_case/breast_cancer_operator_case.json')
    reduction = _load_json_rel('reports/real_reduction_example/breast_cancer_case.json')
    bcw = _load_json_rel('reports/datasets/breast_cancer/summary.json')
    calibration = _load_json_rel('reports/datasets/breast_cancer/calibration.json')
    synthetic_native = _load_json_rel('reports/datasets/synthetic_ruptures/baseline_comparison_native.json')
    synthetic_equal = _load_json_rel('reports/datasets/synthetic_ruptures/baseline_comparison_equal_guardrail.json')
    structure = _load_json_rel('reports/structure_aware_benchmark/breast_cancer.json')
    structure_wine = _load_json_rel('reports/structure_aware_benchmark/wine_risk.json')
    structure_diabetes = _load_json_rel('reports/structure_aware_benchmark/diabetes_binary.json')
    ablation = _load_json_rel('reports/datasets/breast_cancer/ablation.json')

    def _pick_policy(payload: dict[str, Any], name: str) -> dict[str, Any]:
        for row in payload.get('rows', []):
            if row.get('policy') == name:
                return row
        return {}

    md: list[str] = [
        '# Воспроизводимые результаты для вставки в статью',
        '',
        'Ниже приведён единый набор численных результатов и артефактов, полученных из программной реализации. '
        'Все значения взяты из машинно-читаемых отчётов репозитория; `agreement_proxy` означает совпадение с proxy-правилом, а не экспертную клиническую разметку.',
        '',
        '## Среда и ExplainPlan',
        '',
        _row(['Параметр', 'Значение']),
        _row(['---', '---']),
        _row(['Python', f"`{payload['environment']['python']}`"]),
        _row(['Platform', f"`{payload['environment']['platform']}`"]),
        _row(['ExplainPlan SHA256', f"`{payload['explain_plan']['canonical_sha256']}`"]),
        _row(['Mode', f"`{payload['explain_plan']['canonical_json']['mode']}`"]),
        _row(['Risk weights', f"`{payload['explain_plan']['canonical_json']['w_p']}, {payload['explain_plan']['canonical_json']['w_u']}, {payload['explain_plan']['canonical_json']['w_I']}, {payload['explain_plan']['canonical_json']['w_Delta']}, {payload['explain_plan']['canonical_json']['w_R']}`"]),
        _row(['Thresholds', f"`{payload['explain_plan']['canonical_json']['theta_1']}, {payload['explain_plan']['canonical_json']['theta_2']}, {payload['explain_plan']['canonical_json']['theta_3']}, {payload['explain_plan']['canonical_json']['theta_4']}`"]),
        _row(['gamma_max / I_min / Delta_max', f"`{payload['explain_plan']['canonical_json']['gamma_max']} / {payload['explain_plan']['canonical_json']['i_min']} / {payload['explain_plan']['canonical_json']['delta_max']}`"]),
        '',
        '## Сквозной операторный пример sample_113',
        '',
        _row(['Поле', 'Значение']),
        _row(['---', '---']),
        _row(['sample_id', f"`{operator.get('sample_id', 'N/A')}`"]),
        _row(['P(malignant)', _fmt(operator.get('P(malignant)'), 6)]),
        _row(['selected_features', f"`{', '.join(operator.get('selected_features', []))}`"]),
        _row(['mu_low / mu_medium / mu_high', f"`{_fmt(operator.get('mu_low'), 6)} / {_fmt(operator.get('mu_medium'), 6)} / {_fmt(operator.get('mu_high'), 6)}`"]),
        _row(['active_rules', f"`{operator.get('active_rules', [])}`"]),
        _row(['U_model / U_rules / U_trace', f"`{_fmt(operator.get('U_model'), 6)} / {_fmt(operator.get('U_rules'), 6)} / {_fmt(operator.get('U_trace'), 6)}`"]),
        _row(['u_M / I_pre', f"`{_fmt(operator.get('u_M'), 6)} / {_fmt(operator.get('I_pre'), 6)}`"]),
        _row(['gamma_model_to_risk', _fmt(operator.get('gamma_model_to_risk'), 6)]),
        _row(['chi_R / chi_R_crit / action', f"`{operator.get('chi_R')} / {operator.get('chi_R_crit')} / {operator.get('action')}`"]),
        '',
        '## Редукция и риск-наблюдатель sample_113',
        '',
        _row(['Поле', 'Значение']),
        _row(['---', '---']),
        _row(['selected_representation', f"`{reduction.get('selected_representation', reduction.get('selected_class', 'N/A'))}`"]),
        _row(['Delta / I_pre / rho', f"`{_fmt(reduction.get('reduction_loss', reduction.get('Delta')), 6)} / {_fmt(reduction.get('I_pre'), 6)} / {_fmt(reduction.get('rho'), 6)}`"]),
        _row(['chi_Auto / chi_R / chi_R_crit', f"`{_fmt(reduction.get('chi_Auto'))} / {reduction.get('chi_R')} / {reduction.get('chi_R_crit')}`"]),
        _row(['action / reason', f"`{reduction.get('action')}` / `{reduction.get('risk_breakdown', {}).get('reason', '')}`"]),
        '',
        '## Breast Cancer Wisconsin: количественная проверка',
        '',
        _row(['Метрика', 'Значение']),
        _row(['---', '---:']),
        _row(['n', bcw.get('n', 'N/A')]),
        _row(['model_accuracy', _fmt(bcw.get('model_accuracy'), 4)]),
        _row(['model_roc_auc', _fmt(bcw.get('model_roc_auc'), 4)]),
        _row(['model_f1 / precision / recall', f"`{_fmt(bcw.get('model_f1'), 4)} / {_fmt(bcw.get('model_precision'), 4)} / {_fmt(bcw.get('model_recall'), 4)}`"]),
        _row(['agreement_proxy', _fmt(bcw.get('agreement_proxy'), 4)]),
        _row(['missed_critical_ruptures', bcw.get('missed_critical_ruptures', 'N/A')]),
        _row(['false_auto_accept_rate', _fmt(bcw.get('false_auto_accept_rate'), 4)]),
        _row(['mean_I_pre / mean_rho', f"`{_fmt(bcw.get('mean_I_pre'), 4)} / {_fmt(bcw.get('mean_rho'), 4)}`"]),
        '',
        '## Распределения I_pre и rho',
        '',
        _row(['Показатель', 'mean', 'std', 'median', 'p25', 'p75', 'p05', 'p95']),
        _row(['---', '---:', '---:', '---:', '---:', '---:', '---:', '---:']),
        _row(['I_pre', _fmt(bcw.get('i_pre_mean')), _fmt(bcw.get('i_pre_std')), _fmt(bcw.get('i_pre_median')), _fmt(bcw.get('i_pre_p25')), _fmt(bcw.get('i_pre_p75')), _fmt(bcw.get('i_pre_p05')), _fmt(bcw.get('i_pre_p95'))]),
        _row(['rho', _fmt(bcw.get('rho_mean')), _fmt(bcw.get('rho_std')), _fmt(bcw.get('rho_median')), _fmt(bcw.get('rho_p25')), _fmt(bcw.get('rho_p75')), _fmt(bcw.get('rho_p05')), _fmt(bcw.get('rho_p95'))]),
        '',
        '## Калибровка риск-наблюдателя',
        '',
        _row(['Режим', 'agreement_proxy', 'agreement_reference', 'missed_critical', 'critical_recall', 'false_auto_accept']),
        _row(['---', '---:', '---:', '---:', '---:', '---:']),
    ]
    before = calibration.get('before_calibration', {})
    after = calibration.get('after_calibration', {})
    md.append(_row(['before', _fmt(before.get('agreement_proxy')), _fmt(before.get('agreement_reference')), before.get('missed_critical_ruptures', 'N/A'), _fmt(before.get('critical_rupture_recall')), _fmt(before.get('false_auto_accept_rate'))]))
    md.append(_row(['after', _fmt(after.get('agreement_proxy')), _fmt(after.get('agreement_reference')), after.get('missed_critical_ruptures', 'N/A'), _fmt(after.get('critical_rupture_recall')), _fmt(after.get('false_auto_accept_rate'))]))
    best = calibration.get('best_params', {})
    md += [
        '',
        f"Калибровочная цель: `{calibration.get('objective', '')}`.",
        f"Лучшие параметры: weights=`{best.get('weights', {})}`, thresholds=`{best.get('thresholds', [])}`, gamma_max=`{best.get('gamma_max')}`, I_min=`{best.get('i_min')}`, Delta_max=`{best.get('delta_max')}`.",
        '',
        '## Synthetic ruptures: native vs equal-guardrail',
        '',
        'В режиме `native` baseline-методы получают только собственные входы и не получают `chi_R_crit`; полный наблюдатель получает полный структурный вход.',
        '',
        _row(['Метод', 'access', 'agreement_ref', 'missed_critical', 'critical_recall', 'false_auto_accept', 'auto_accept_cov']),
        _row(['---', '---', '---:', '---:', '---:', '---:', '---:']),
    ]
    for row in synthetic_native.get('rows', []):
        md.append(_row([row.get('baseline'), row.get('information_access'), _fmt(row.get('agreement_reference')), row.get('missed_critical_ruptures'), _fmt(row.get('critical_rupture_recall')), _fmt(row.get('false_auto_accept_rate')), _fmt(row.get('auto_accept_coverage'))]))
    md += [
        '',
        'В режиме `equal_guardrail` всем методам передаётся внешний safety-флаг; это sanity-check политики блокировки.',
        '',
        _row(['Метод', 'access', 'agreement_ref', 'missed_critical', 'critical_recall', 'false_auto_accept']),
        _row(['---', '---', '---:', '---:', '---:', '---:']),
    ]
    for row in synthetic_equal.get('rows', []):
        md.append(_row([row.get('baseline'), row.get('information_access'), _fmt(row.get('agreement_reference')), row.get('missed_critical_ruptures'), _fmt(row.get('critical_rupture_recall')), _fmt(row.get('false_auto_accept_rate'))]))
    md += [
        '',
        '## Structure-aware benchmark',
        '',
        f"Сценарии: `{', '.join(structure.get('scenarios', []))}`; число кейсов: `{structure.get('n_cases', 'N/A')}`.",
        '',
        _row(['Policy', 'agreement_ref', 'missed_critical', 'critical_recall', 'false_auto_accept', 'auto_accept_cov']),
        _row(['---', '---:', '---:', '---:', '---:', '---:']),
    ]
    for row in structure.get('rows', []):
        md.append(_row([row.get('policy'), _fmt(row.get('agreement_reference')), row.get('missed_critical_ruptures'), _fmt(row.get('critical_rupture_recall')), _fmt(row.get('false_auto_accept_rate')), _fmt(row.get('auto_accept_coverage'))]))
    md += [
        '',
        '## Улучшения не только на синтетике (real rows + structure-aware)',
        '',
        _row(['Dataset', 'full_agreement_ref', 'threshold_agreement_ref', 'agreement_gain', 'full_false_auto_accept', 'threshold_false_auto_accept', 'false_auto_accept_drop']),
        _row(['---', '---:', '---:', '---:', '---:', '---:', '---:']),
    ]
    for ds_name, payload in (
        ('breast_cancer', structure),
        ('wine_risk', structure_wine),
        ('diabetes_binary', structure_diabetes),
    ):
        full = _pick_policy(payload, 'full_observer_calibrated')
        thr = _pick_policy(payload, 'probability_threshold')
        if not full or not thr:
            continue
        fa_full = float(full.get('false_auto_accept_rate', 0.0))
        fa_thr = float(thr.get('false_auto_accept_rate', 0.0))
        ag_full = float(full.get('agreement_reference', 0.0))
        ag_thr = float(thr.get('agreement_reference', 0.0))
        md.append(_row([
            ds_name,
            _fmt(ag_full),
            _fmt(ag_thr),
            _fmt(ag_full - ag_thr),
            _fmt(fa_full),
            _fmt(fa_thr),
            _fmt(fa_thr - fa_full),
        ]))
    md += [
        '',
        '## Абляционный анализ',
        '',
        _row(['Mode', 'agreement_proxy', 'missed_critical', 'critical_recall', 'false_auto_accept', 'auto_accept_cov', 'mean_reduction_loss']),
        _row(['---', '---:', '---:', '---:', '---:', '---:', '---:']),
    ]
    for row in ablation.get('rows', []):
        md.append(_row([row.get('mode'), _fmt(row.get('agreement_proxy')), row.get('missed_critical_ruptures'), _fmt(row.get('critical_rupture_recall')), _fmt(row.get('false_auto_accept_rate')), _fmt(row.get('auto_accept_coverage')), _fmt(row.get('mean_reduction_loss'))]))
    md += [
        '',
        '## Команды воспроизведения',
        '',
    ]
    md.extend([f"- `{cmd}`" for cmd in payload.get('commands', [])])
    md += [
        '',
        '## Артефакты и SHA256',
        '',
        _row(['Артефакт', 'Глава', 'SHA256', 'Что подтверждает']),
        _row(['---', '---', '---', '---']),
    ]
    for row in payload.get('artifacts', []):
        md.append(_row([f"`{row['path']}`", row['chapter'], f"`{row['sha256']}`", row['confirms']]))
    md += [
        '',
        '## Корректная формулировка для текста',
        '',
        'Встроенные наборы данных используются для количественной проверки и калибровки наблюдающего контура; '
        'внешние registry-режимы подтверждают переносимость пайплайна; диагностический режим `synthetic_ruptures` проверяет safety-свойство: '
        '`chi_R^crit=1` приводит к `block`, а в native-режиме baseline-методы без доступа к структурному индикатору пропускают критические разрывы.',
        '',
    ]
    return '\n'.join(md)


def build_chapter5_ready(payload: dict[str, Any]) -> str:
    bcw = _load_json_rel('reports/datasets/breast_cancer/summary.json')
    calibration = _load_json_rel('reports/datasets/breast_cancer/calibration.json')
    synthetic_native = _load_json_rel('reports/datasets/synthetic_ruptures/baseline_comparison_native.json')
    synthetic_equal = _load_json_rel('reports/datasets/synthetic_ruptures/baseline_comparison_equal_guardrail.json')
    structure = _load_json_rel('reports/structure_aware_benchmark/breast_cancer.json')
    structure_wine = _load_json_rel('reports/structure_aware_benchmark/wine_risk.json')
    structure_diabetes = _load_json_rel('reports/structure_aware_benchmark/diabetes_binary.json')
    ablation = _load_json_rel('reports/datasets/breast_cancer/ablation.json')

    def _pick_policy(payload: dict[str, Any], name: str) -> dict[str, Any]:
        for row in payload.get('rows', []):
            if row.get('policy') == name:
                return row
        return {}

    md: list[str] = [
        '# Глава 5. Экспериментальная проверка и воспроизводимые артефакты',
        '',
        'В данной главе описана экспериментальная проверка программной реализации наблюдающего контура FuzzyXAI. '
        'Цель экспериментов состоит не только в оценке качества базовой модели, но и в проверке safety-свойств: '
        'корректной блокировки критических разрывов, влияния контекстного слоя, роли редукции представлений и воспроизводимости расчётов.',
        '',
        '## 5.1. Единая демонстрационная среда',
        '',
        'Интерактивная реализация сведена в единую среду `FuzzyXAI Studio`. В одном интерфейсе отображается маршрут '
        '`Dataset -> E_k -> A_k^F -> chi_Auto -> rho -> Action`, что позволяет проверять не отдельные скрипты, а полный контур принятия решения.',
        '',
        '![Рисунок 5.1. Главный экран FuzzyXAI Studio](../browser_visual_check/01_studio_home.png)',
        '',
        'Рисунок 5.1 показывает общий экран: выбор датасета и сценария, итоговое действие, функцию принадлежности, вклад компонентов риска, распределение риска по датасету и маршрут pipeline.',
        '',
        '![Рисунок 5.2. Панель What-if для проверки действия политики](../browser_visual_check/04_what_if.png)',
        '',
        'Рисунок 5.2 иллюстрирует интерактивную проверку параметров риска, интерпретируемости, потери редукции, `chi_R`, `chi_R^crit`, `chi_Auto` и trace-состояния.',
        '',
        '![Рисунок 5.3. След операторов и источники данных](../browser_visual_check/08_operator_trace.png)',
        '',
        'Рисунок 5.3 показывает, какие операторы используются в цепочке и какие поля системы они читают: модельный прогноз, объяснительный объект, контекст, путь/разрыв и действие наблюдателя.',
        '',
        '![Рисунок 5.4. Граф операторного маршрута](../browser_visual_check/09_operator_flow.png)',
        '',
        'Рисунок 5.4 визуализирует последовательность операторов и диагностические состояния переходов.',
        '',
        '## 5.2. Датасеты и роли режимов',
        '',
        _row(['Режим данных', 'Роль в эксперименте', 'Интерпретация результата']),
        _row(['---', '---', '---']),
        _row(['breast_cancer', 'количественная медицинская проверка', 'проверка модели, наблюдателя, I_pre, rho и калибровки']),
        _row(['wine_risk', 'табличная переносимость', 'проверка применимости контура вне медицинского набора']),
        _row(['synthetic_ruptures', 'diagnostic/safety стенд', 'контролируемые критические разрывы и проверка chi_R^crit -> block']),
        _row(['diabetes_binary', 'stress-test калибровки', 'пограничная неопределенность и чувствительность политики']),
        _row(['registry_*', 'external-transfer/readiness', 'подключение внешних предметных источников без claims о качестве модели']),
        '',
        '## 5.3. Breast Cancer Wisconsin',
        '',
        _row(['Метрика', 'Значение']),
        _row(['---', '---:']),
        _row(['n', bcw.get('n', 'N/A')]),
        _row(['model_accuracy', _fmt(bcw.get('model_accuracy'), 4)]),
        _row(['model_roc_auc', _fmt(bcw.get('model_roc_auc'), 4)]),
        _row(['model_f1', _fmt(bcw.get('model_f1'), 4)]),
        _row(['precision', _fmt(bcw.get('model_precision'), 4)]),
        _row(['recall', _fmt(bcw.get('model_recall'), 4)]),
        _row(['agreement_proxy', _fmt(bcw.get('agreement_proxy'), 4)]),
        _row(['missed_critical_ruptures', bcw.get('missed_critical_ruptures', 'N/A')]),
        _row(['false_auto_accept_rate', _fmt(bcw.get('false_auto_accept_rate'), 4)]),
        '',
        'На Breast Cancer Wisconsin вероятностный baseline является сильным, поскольку reference-политика существенно зависит от вероятности модели. Поэтому этот режим используется как проверка воспроизводимости полного контура и калибровки, а не как единственное доказательство преимущества над risk-only baseline.',
        '',
        '## 5.4. Распределения I_pre и rho',
        '',
        _row(['Показатель', 'mean', 'std', 'median', 'p25', 'p75', 'p05', 'p95']),
        _row(['---', '---:', '---:', '---:', '---:', '---:', '---:', '---:']),
        _row(['I_pre', _fmt(bcw.get('i_pre_mean')), _fmt(bcw.get('i_pre_std')), _fmt(bcw.get('i_pre_median')), _fmt(bcw.get('i_pre_p25')), _fmt(bcw.get('i_pre_p75')), _fmt(bcw.get('i_pre_p05')), _fmt(bcw.get('i_pre_p95'))]),
        _row(['rho', _fmt(bcw.get('rho_mean')), _fmt(bcw.get('rho_std')), _fmt(bcw.get('rho_median')), _fmt(bcw.get('rho_p25')), _fmt(bcw.get('rho_p75')), _fmt(bcw.get('rho_p05')), _fmt(bcw.get('rho_p95'))]),
        '',
        '## 5.5. Калибровка наблюдателя',
        '',
        'Калибровка выполняется на validation-разбиении и оценивается отдельно; прогнозная модель при этом не изменяется.',
        '',
        _row(['Режим', 'agreement_proxy', 'agreement_reference', 'missed_critical', 'critical_recall', 'false_auto_accept']),
        _row(['---', '---:', '---:', '---:', '---:', '---:']),
    ]
    before = calibration.get('before_calibration', {})
    after = calibration.get('after_calibration', {})
    md.append(_row(['до калибровки', _fmt(before.get('agreement_proxy')), _fmt(before.get('agreement_reference')), before.get('missed_critical_ruptures', 'N/A'), _fmt(before.get('critical_rupture_recall')), _fmt(before.get('false_auto_accept_rate'))]))
    md.append(_row(['после калибровки', _fmt(after.get('agreement_proxy')), _fmt(after.get('agreement_reference')), after.get('missed_critical_ruptures', 'N/A'), _fmt(after.get('critical_rupture_recall')), _fmt(after.get('false_auto_accept_rate'))]))
    best = calibration.get('best_params', {})
    md += [
        '',
        f"Целевая функция калибровки: `{calibration.get('objective', '')}`.",
        f"Лучшие параметры: weights=`{best.get('weights', {})}`, thresholds=`{best.get('thresholds', [])}`, gamma_max=`{best.get('gamma_max')}`, I_min=`{best.get('i_min')}`, Delta_max=`{best.get('delta_max')}`.",
        '',
        '## 5.6. Diagnostic safety: synthetic_ruptures',
        '',
        'Для проверки диагностического слоя используется два режима доступа. В `native` baseline-методы не получают `chi_R^crit`; в `equal_guardrail` всем методам передаётся внешний safety-флаг. Поэтому `equal_guardrail` служит sanity-check, а научное сравнение проводится в `native`.',
        '',
        '![Рисунок 5.5. Benchmark-панель Studio](../browser_visual_check/05_benchmark.png)',
        '',
        _row(['Метод', 'access', 'agreement_ref', 'missed_critical', 'critical_recall', 'false_auto_accept']),
        _row(['---', '---', '---:', '---:', '---:', '---:']),
    ]
    for row in synthetic_native.get('rows', []):
        md.append(_row([row.get('baseline'), row.get('information_access'), _fmt(row.get('agreement_reference')), row.get('missed_critical_ruptures'), _fmt(row.get('critical_rupture_recall')), _fmt(row.get('false_auto_accept_rate'))]))
    md += [
        '',
        'В `native`-режиме полный наблюдатель не пропускает критические разрывы (`missed_critical_ruptures=0`, `critical_rupture_recall=1.0`), тогда как risk-only и XAI-baseline без структурного индикатора пропускают 70 критических случаев.',
        '',
        '### Equal-guardrail sanity-check',
        '',
        _row(['Метод', 'access', 'agreement_ref', 'missed_critical', 'critical_recall', 'false_auto_accept']),
        _row(['---', '---', '---:', '---:', '---:', '---:']),
    ]
    for row in synthetic_equal.get('rows', []):
        md.append(_row([row.get('baseline'), row.get('information_access'), _fmt(row.get('agreement_reference')), row.get('missed_critical_ruptures'), _fmt(row.get('critical_rupture_recall')), _fmt(row.get('false_auto_accept_rate'))]))
    md += [
        '',
        '## 5.7. Structure-aware benchmark',
        '',
        f"Эксперимент использует реальные строки Breast Cancer Wisconsin и контролируемые perturbation-сценарии: `{', '.join(structure.get('scenarios', []))}`. Число кейсов: `{structure.get('n_cases', 'N/A')}`.",
        '',
        _row(['Policy', 'agreement_ref', 'missed_critical', 'critical_recall', 'false_auto_accept', 'auto_accept_cov']),
        _row(['---', '---:', '---:', '---:', '---:', '---:']),
    ]
    for row in structure.get('rows', []):
        md.append(_row([row.get('policy'), _fmt(row.get('agreement_reference')), row.get('missed_critical_ruptures'), _fmt(row.get('critical_rupture_recall')), _fmt(row.get('false_auto_accept_rate')), _fmt(row.get('auto_accept_coverage'))]))
    md += [
        '',
        '## 5.7a. Несинтетические улучшения на реальных строках',
        '',
        'Ниже сводка structure-aware прогонов на реальных датасетах (с контролируемыми perturbation-сценариями). '
        'Показаны прирост `agreement_reference` и снижение `false_auto_accept_rate` относительно risk-only threshold.',
        '',
        _row(['Dataset', 'full_agreement_ref', 'threshold_agreement_ref', 'agreement_gain', 'full_false_auto_accept', 'threshold_false_auto_accept', 'false_auto_accept_drop']),
        _row(['---', '---:', '---:', '---:', '---:', '---:', '---:']),
    ]
    for ds_name, payload in (
        ('breast_cancer', structure),
        ('wine_risk', structure_wine),
        ('diabetes_binary', structure_diabetes),
    ):
        full = _pick_policy(payload, 'full_observer_calibrated')
        thr = _pick_policy(payload, 'probability_threshold')
        if not full or not thr:
            continue
        fa_full = float(full.get('false_auto_accept_rate', 0.0))
        fa_thr = float(thr.get('false_auto_accept_rate', 0.0))
        ag_full = float(full.get('agreement_reference', 0.0))
        ag_thr = float(thr.get('agreement_reference', 0.0))
        md.append(_row([
            ds_name,
            _fmt(ag_full),
            _fmt(ag_thr),
            _fmt(ag_full - ag_thr),
            _fmt(fa_full),
            _fmt(fa_thr),
            _fmt(fa_thr - fa_full),
        ]))
    md += [
        '',
        '## 5.8. Абляционный анализ',
        '',
        _row(['Mode', 'agreement_proxy', 'missed_critical', 'critical_recall', 'false_auto_accept', 'auto_accept_cov', 'mean_reduction_loss']),
        _row(['---', '---:', '---:', '---:', '---:', '---:', '---:']),
    ]
    for row in ablation.get('rows', []):
        md.append(_row([row.get('mode'), _fmt(row.get('agreement_proxy')), row.get('missed_critical_ruptures'), _fmt(row.get('critical_rupture_recall')), _fmt(row.get('false_auto_accept_rate')), _fmt(row.get('auto_accept_coverage')), _fmt(row.get('mean_reduction_loss'))]))
    md += [
        '',
        'Абляция показывает, что отключение контекстного слоя (`no_topos`) увеличивает долю ложного автоматического применения, а переход к `f0_only` повышает среднюю потерю редукции. Простое пороговое правило по вероятности имеет высокую proxy-согласованность на risk-dominated разметке, но даёт более высокую долю автоматического применения и не проверяет структурные ограничения.',
        '',
        '## 5.9. Воспроизводимость',
        '',
        _row(['Команда', 'Назначение']),
        _row(['---', '---']),
        _row(['`make dissertation-check`', 'полная проверка тестов, датасетов, отчётов и карточек']),
        _row(['`make thesis-practice-tables`', 'пересборка таблиц для диссертации/статьи']),
        _row(['`make browser-visual-check`', 'актуальные скриншоты GUI через Chromium']),
        _row(['`make ui-health-check`', 'smoke-check единого GUI']),
        _row(['`make reproducibility-artifacts`', 'manifest, ExplainPlan hash и готовые материалы для текста']),
        '',
        'Ключевые артефакты и SHA256 приведены в `reports/reproducibility_artifacts/manifest.md` и `reports/reproducibility_artifacts/article_insert.md`.',
        '',
        '## 5.10. Вывод по главе',
        '',
        'Экспериментальная часть показывает, что предложенный контур воспроизводимо строит объяснительные объекты, вычисляет метрики интерпретируемости и риска, выбирает представление неопределённости и применяет safety-политику. Основное преимущество метода проявляется не в превосходстве над risk-only baseline на каждом табличном наборе, а в структуральной проверке условий автоматического применения: критический разрыв `chi_R^crit=1` приводит к `block`, контекстный слой ограничивает `accept`, а редукция представлений контролируется через `Delta`.',
        '',
    ]
    return '\n'.join(md)

def write_reports(payload: dict[str, Any], out_dir: str | Path) -> dict[str, str]:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    plan_path = out / 'explain_plan.json'
    plan_path.write_text(json.dumps(payload['explain_plan']['canonical_json'], ensure_ascii=False, indent=2), encoding='utf-8')

    hash_path = out / 'explain_plan.sha256'
    hash_path.write_text(str(payload['explain_plan']['canonical_sha256']) + '\n', encoding='utf-8')

    for row in payload['artifacts']:
        p = ROOT / str(row['path'])
        row['exists'] = p.exists()
        row['sha256'] = _sha256_file(p)

    json_path = out / 'manifest.json'
    json_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding='utf-8')

    csv_path = out / 'manifest.csv'
    rows = payload['artifacts']
    with csv_path.open('w', encoding='utf-8', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=['path', 'exists', 'sha256', 'chapter', 'confirms', 'command'])
        writer.writeheader()
        writer.writerows(rows)

    md_path = out / 'manifest.md'
    md = [
        '# Reproducibility Artifacts',
        '',
        f"- status: `{payload['status']}`",
        f"- python: `{payload['environment']['python']}`",
        f"- platform: `{payload['environment']['platform']}`",
        f"- ExplainPlan SHA256: `{payload['explain_plan']['canonical_sha256']}`",
        '',
        '## Commands',
        '',
    ]
    md.extend([f"- `{cmd}`" for cmd in payload['commands']])
    md += [
        '',
        '## Artifact Manifest',
        '',
        '| artifact | exists | sha256 | chapter | confirms | command |',
        '| --- | --- | --- | --- | --- | --- |',
    ]
    for row in rows:
        sha = row['sha256'] or ''
        md.append(
            f"| `{row['path']}` | `{row['exists']}` | `{sha[:12]}` | {row['chapter']} | {row['confirms']} | `{row['command']}` |"
        )
    md_path.write_text('\n'.join(md), encoding='utf-8')

    article_path = out / 'article_insert.md'
    article_path.write_text(build_article_insert(payload), encoding='utf-8')

    chapter5_dir = ROOT / 'reports' / 'chapter5'
    chapter5_dir.mkdir(parents=True, exist_ok=True)
    chapter5_path = chapter5_dir / 'chapter5_ready_with_figures.md'
    chapter5_path.write_text(build_chapter5_ready(payload), encoding='utf-8')

    return {
        'plan': str(plan_path),
        'hash': str(hash_path),
        'json': str(json_path),
        'csv': str(csv_path),
        'md': str(md_path),
        'article_insert': str(article_path),
        'chapter5_ready': str(chapter5_path),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--out-dir', default='reports/reproducibility_artifacts')
    args = parser.parse_args()
    payload = build_payload()
    paths = write_reports(payload, args.out_dir)
    print(json.dumps({'status': 'ok', 'paths': paths}, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
