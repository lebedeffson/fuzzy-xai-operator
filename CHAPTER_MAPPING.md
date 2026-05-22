# Связь глав диссертации и реализации

## Глава 2

| Объект диссертации | Реализация | Проверка |
|---|---|---|
| Объект объяснения `E_k` | `fuzzyxai/core/explanation_object.py` | `tests/test_composition.py` |
| `ExplainPlan` | `fuzzyxai/core/explain_plan.py` | `tests/test_v3_pipeline.py` |
| Генерация плана из данных | `fuzzyxai/core/plan_builder.py` | `tests/test_v8_synthetic_gui_support.py` |
| Системный оператор `Omega` | `fuzzyxai/core/system_operator.py` | `proofs/chapter2_operator_proof.py` |
| Композиция `E_i o E_j` | `fuzzyxai/core/composition.py` | `proofs/chapter2_operator_proof.py` |
| Семантическое рассогласование `d_E` | `fuzzyxai/core/trust_evaluator.py` | `reports/chapter2_proof.json` |
| Потеря и индекс `L(E), I(E)` | `fuzzyxai/core/trust_evaluator.py` | `reports/chapter2_proof.json` |
| Диагностика `D_ij` | `fuzzyxai/core/diagnostics.py` | `tests/test_composition.py` |
| Калибровка `beta` | `fuzzyxai/calibration/` | `proofs/chapter2_calibration_proof.py` |
| Визуализация композиции | `fuzzyxai/visual/` | `apps/defense_demo.py` |

## Глава 3

| Объект диссертации | Реализация | Проверка |
|---|---|---|
| Классическое представление `F0` | `fuzzyxai/hierarchy/f0.py` | `tests/test_reductions.py` |
| Интервальное представление `F_I` | `fuzzyxai/hierarchy/interval.py` | `tests/test_reductions.py` |
| Hesitant-представление `F_H` | `fuzzyxai/hierarchy/hesitant.py` | `tests/test_reductions.py` |
| Нейтрософское source-aware представление | `fuzzyxai/hierarchy/neutrosophic.py` | `tests/test_reductions.py` |
| Декоратор источника | `fuzzyxai/hierarchy/source_annotation.py` | hierarchy API |
| Многоуровневое представление `F_ML` | `fuzzyxai/hierarchy/multilevel.py` | `examples/chapter3_end_to_end.py` |
| Редукции и `Delta` | `fuzzyxai/hierarchy/reductions.py`, `meta_reducer.py` | `reports/chapter3_end_to_end_report.json` |
| Профиль `P_sit` | `fuzzyxai/selection/profile_builder.py` | `proofs/chapter3_hierarchy_proof.py` |
| Парето-выбор класса | `fuzzyxai/selection/pareto_selector.py` | `tests/test_selector.py` |
| Диагностика выбора `D_choice` | `fuzzyxai/selection/choice_diagnostic.py` | `reports/chapter3_end_to_end_report.json` |
| Совместимость и FML-синтез | `fuzzyxai/selection/compatibility.py` | `reports/chapter3_end_to_end_report.json` |

## Сквозная связь глав 2 и 3

| Связь | Реализация | Проверка |
|---|---|---|
| Замена `mu_k` на `A_k^F` | `ExplanationObject.representation` | `examples/chapter3_end_to_end.py` |
| Потеря редукции в `d_E_ext` | `semantic_disagreement(... beta['reduction'])` | `reports/chapter3_end_to_end_report.json` |
| Композиция расширенных объектов | `compose(e_ext, e_dec, beta)` | `examples/chapter3_end_to_end.py` |
| Сравнение “без оператора / с оператором” | `benchmarks/operator_comparison_benchmark.py` | `tests/test_operator_comparison_benchmark.py` |

## Risk-Aware Observer

| Компонент | Реализация | Проверка |
|---|---|---|
| Модельная неопределённость | `fuzzyxai/risk/uncertainty.py` | `tests/test_risk_uncertainty.py` |
| Политика безопасного действия | `fuzzyxai/risk/decision_policy.py` | `tests/test_decision_policy.py` |
| Обёртка над моделью | `fuzzyxai/risk/risk_aware_model.py` | `tests/test_risk_aware_model.py` |
| Метрики покрытия, стоимости и риска | `fuzzyxai/risk/metrics.py` | `tests/test_risk_metrics.py` |
| Benchmark наблюдателя | `benchmarks/risk_aware_observer_benchmark.py` | `tests/test_risk_aware_benchmark.py` |
| GUI-раздел наблюдателя | `apps/defense_demo.py` | `tests/test_defense_demo.py` |

## Финальные артефакты проверки

- Численный маршрут главы 2 (`mu`, `H`, `d_E`, композиция, `L`, `I`, `D_ij`) -> `proofs/validate_thesis_examples.py` -> `reports/thesis_validation.json`.
- Численный маршрут главы 3 (`P_sit`, выбор класса, редукции, `Delta`, `d_E_ext`, `D_choice`) -> `proofs/validate_thesis_examples.py` -> `reports/thesis_validation.json`.
- Сквозная defense demo -> `examples/thesis_demo.py` -> `reports/thesis_demo_report.json` и `reports/thesis_demo_composition_graph.html`.
- Risk-Aware Observer benchmark -> `benchmarks/risk_aware_observer_benchmark.py` -> `reports/risk_aware_observer_benchmark.json` и `reports/risk_aware_observer_benchmark.md`.
