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
| Полный маршрут “данные -> модель -> глава 2 -> глава 3 -> наблюдатель” | `full_pipeline_demo.py` | `tests/test_full_pipeline_demo.py` |

## Risk-Aware Observer

| Компонент | Реализация | Проверка |
|---|---|---|
| Модельная неопределённость | `fuzzyxai/risk/uncertainty.py` | `tests/test_risk_uncertainty.py` |
| Политика безопасного действия | `fuzzyxai/risk/decision_policy.py` | `tests/test_decision_policy.py` |
| Обёртка над моделью | `fuzzyxai/risk/risk_aware_model.py` | `tests/test_risk_aware_model.py` |
| Метрики покрытия, стоимости и риска | `fuzzyxai/risk/metrics.py` | `tests/test_risk_metrics.py` |
| Benchmark наблюдателя | `benchmarks/risk_aware_observer_benchmark.py` | `tests/test_risk_aware_benchmark.py` |
| GUI-раздел наблюдателя | `apps/defense_demo.py` | `tests/test_defense_demo.py` |

## Active Risk-Aware XAI Observer

| Компонент | Реализация | Проверка |
|---|---|---|
| Расширенный объект `E_M^ext` над прогнозным интерфейсом | `fuzzyxai/risk/observer_pipeline.py` | `tests/test_full_observer_pipeline.py` |
| Риск автоматического применения `rho(x)` | `RiskPolicy.risk_score` | `tests/test_full_observer_pipeline.py` |
| Композиция `E_D o E_R o E_M^ext` | `ObserverPipeline.explain_case` | `reports/full_observer_pipeline/full_observer_pipeline.md` |
| Исполняемый сценарий наблюдателя | `full_observer_pipeline.py` | `make full-observer` |

## LOFO-F1 Rule Pruning

| Компонент | Реализация | Проверка |
|---|---|---|
| Leave-One-Rule-Out F1 importance | `fuzzyxai/rules/lofo_f1.py` | `tests/test_lofo_f1_rule_pruning.py` |
| Bootstrap-стабилизация top-B | `bootstrap_lofo_f1_importance` | `tests/test_lofo_f1_rule_pruning.py` |
| Сравнение с Budget-Prune | `benchmarks/lofo_f1_rule_pruning_demo.py` | `reports/lofo_f1_rule_pruning.md` |

## Финальные артефакты проверки

- Численный маршрут главы 2 (`mu`, `H`, `d_E`, композиция, `L`, `I`, `D_ij`) -> `proofs/validate_thesis_examples.py` -> `reports/thesis_validation.json`.
- Численный маршрут главы 3 (`P_sit`, выбор класса, редукции, `Delta`, `d_E_ext`, `D_choice`) -> `proofs/validate_thesis_examples.py` -> `reports/thesis_validation.json`.
- Сквозная defense demo -> `examples/thesis_demo.py` -> `reports/thesis_demo_report.json` и `reports/thesis_demo_composition_graph.html`.
- Risk-Aware Observer benchmark -> `benchmarks/risk_aware_observer_benchmark.py` -> `reports/risk_aware_observer_benchmark.json` и `reports/risk_aware_observer_benchmark.md`.
- Полная демонстрация -> `full_pipeline_demo.py` -> `reports/full_demo/index.html`, `reports/full_demo/full_pipeline_report.json`.
- LOFO-F1 rule pruning -> `benchmarks/lofo_f1_rule_pruning_demo.py` -> `reports/lofo_f1_rule_pruning.json` и `reports/lofo_f1_rule_pruning.md`.

- Full observer pipeline -> `full_observer_pipeline.py` -> `reports/full_observer_pipeline/full_observer_pipeline.json`, `reports/full_observer_pipeline/full_observer_pipeline.md`.
