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
| Формальные теоремы главы 2 | `docs/FORMAL_THEOREMS_CH2_CH3_RU.md` | `proofs/formal_theorem_checks.py` |

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
| Формальные теоремы главы 3 | `docs/FORMAL_THEOREMS_CH2_CH3_RU.md` | `proofs/formal_theorem_checks.py` |

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
| Риск автоматического применения `rho(x)` | `fuzzyxai/risk/risk_function.py` | `tests/test_risk_function.py` |
| Оптимальная политика через ожидаемую стоимость | `choose_min_expected_cost` | `tests/test_risk_function.py` |
| Композиция `E_A o E_R o E_M^ext` | `ObserverPipeline.explain_case` | `reports/full_observer_pipeline/full_observer_pipeline.md` |
| Математическое описание слоя | `docs/RISK_AWARE_XAI_OBSERVER_MATH_RU.md` | code-review + tests |
| Слой датасетов и профиля | `fuzzyxai/data/` | `tests/test_dataset_data_layer.py` |
| Dataset observer pipeline | `fuzzyxai/pipelines/dataset_observer_pipeline.py` | `tests/test_dataset_observer_pipeline.py`, `tests/test_dataset_observer_pipeline_correctness.py` |
| Выбор `A_M^F` в dataset pipeline | `fuzzyxai/risk/representation_selection.py` | `tests/test_dataset_observer_representation_integration.py` |
| Объекты `E_R`, `E_A` в отчёте | `fuzzyxai/risk/risk_aware_model.py` | `tests/test_dataset_observer_representation_integration.py` |
| Исполняемый сценарий наблюдателя | `full_observer_pipeline.py` | `make full-observer` |

## LOFO-F1 Rule Pruning

| Компонент | Реализация | Проверка |
|---|---|---|
| Leave-One-Rule-Out F1 importance | `fuzzyxai/rules/lofo_f1.py` | `tests/test_lofo_f1_rule_pruning.py` |
| Bootstrap-стабилизация top-B | `bootstrap_lofo_f1_importance` | `tests/test_lofo_f1_rule_pruning.py` |
| Сравнение с Budget-Prune | `benchmarks/lofo_f1_rule_pruning_demo.py` | `reports/lofo_f1_rule_pruning.md` |

## Категориально-гомотопическое приложение

| Компонент | Реализация | Проверка |
|---|---|---|
| Категория успешных объяснительных согласований `Expl` | `fuzzyxai/category/expl_category.py` | `tests/test_expl_category_laws.py` |
| Морфизмы объяснений | `fuzzyxai/category/morphism.py` | `tests/test_expl_category_laws.py` |
| Диагностическая граница `Expl_D` | `fuzzyxai/category/diagnostic_completion.py` | `tests/test_diagnostic_completion.py` |
| Предпучок контекстов `Set^{Expl^op}` | `fuzzyxai/category/presheaf.py` | `tests/test_presheaf_functoriality.py` |
| Конкретные контексты `Risk/Audit/User/Trace` | `fuzzyxai/category/context_topos.py` | `tests/test_context_topos_smoke.py` |
| Подпредпучок `AutoAccept` | `fuzzyxai/category/subpresheaf.py` | `tests/test_subpresheaf.py` |
| Представимый предпучок Йонеды `y(E)` | `fuzzyxai/category/yoneda.py` | `tests/test_yoneda.py` |
| Дескриптор топоса контекстов | `fuzzyxai/category/context_topos.py` | `tests/test_presheaf_functoriality.py` |
| Контекстная проверка auto-accept | `fuzzyxai/risk/context_acceptance.py` | `tests/test_risk_context_acceptance.py` |
| Тип пути согласования `Path_Expl` | `fuzzyxai/hott/path_type.py` | `tests/test_explanation_path_types.py` |
| Тип разрыва `Rupture` | `fuzzyxai/hott/rupture_type.py` | `tests/test_explanation_path_types.py` |
| Временной дрейф объяснений | `fuzzyxai/hott/drift_path.py` | `tests/test_temporal_drift_paths.py` |
| Сертификаты путей и разрывов | `fuzzyxai/hott/path_certificates.py` | `tests/test_context_topos_smoke.py` |
| Category/HoTT proof report | `proofs/category_hott_checks.py` | `reports/category_hott/category_hott_checks.md` |
| Математическое приложение | `docs/CATEGORICAL_HOTT_EXTENSION_RU.md` | category/hott tests |

## Эксперименты главы 5

| Эксперимент | Реализация | Проверка/отчёт |
|---|---|---|
| Сценарии S0-S6 | `experiments/chapter5_experiments.py` | `reports/chapter5/scenarios_s0_s6.csv` |
| Калибровка весов риска | `calibrate_weights` | `reports/chapter5/chapter5_experiments.md` |
| Сравнение baseline | `baseline_comparison` | `reports/chapter5/baseline_comparison.csv` |
| Чувствительность | `sensitivity` | `reports/chapter5/sensitivity_w_R.html` |
| Вычислительная сложность | `timing_table` | `reports/chapter5/timing_complexity.csv` |
| Breast Cancer validation | `breast_cancer_validation` | `reports/chapter5/breast_cancer_validation.csv` |
| Контекстная логика топоса | `context_logic_table` | `reports/chapter5/context_logic.csv` |

## Финальные артефакты проверки

- Численный маршрут главы 2 (`mu`, `H`, `d_E`, композиция, `L`, `I`, `D_ij`) -> `proofs/validate_thesis_examples.py` -> `reports/thesis_validation.json`.
- Численный маршрут главы 3 (`P_sit`, выбор класса, редукции, `Delta`, `d_E_ext`, `D_choice`) -> `proofs/validate_thesis_examples.py` -> `reports/thesis_validation.json`.
- Сквозная defense demo -> `examples/thesis_demo.py` -> `reports/thesis_demo_report.json` и `reports/thesis_demo_composition_graph.html`.
- Risk-Aware Observer benchmark -> `benchmarks/risk_aware_observer_benchmark.py` -> `reports/risk_aware_observer_benchmark.json` и `reports/risk_aware_observer_benchmark.md`.
- Полная демонстрация -> `full_pipeline_demo.py` -> `reports/full_demo/index.html`, `reports/full_demo/full_pipeline_report.json`.
- LOFO-F1 rule pruning -> `benchmarks/lofo_f1_rule_pruning_demo.py` -> `reports/lofo_f1_rule_pruning.json` и `reports/lofo_f1_rule_pruning.md`.

- Full observer pipeline -> `full_observer_pipeline.py` -> `reports/full_observer_pipeline/full_observer_pipeline.json`, `reports/full_observer_pipeline/full_observer_pipeline.md`.
- Risk-aware observer math -> `docs/RISK_AWARE_XAI_OBSERVER_MATH_RU.md` -> `fuzzyxai/risk/risk_function.py`, `tests/test_risk_function.py`.
- Dataset observer -> `examples/dataset_observer_demo.py` -> `reports/dataset_observer/dataset_observer_report.html`.
- Formal theorem pack -> `docs/FORMAL_THEOREMS_CH2_CH3_RU.md`.
- vNext TZ -> `docs/TZ_MATH_RISK_AWARE_OBSERVER_VNEXT_RU.md`.
- Formal theorem checks -> `proofs/formal_theorem_checks.py` -> `reports/formal_theorems/formal_theorem_checks.md`.
- Implementation inventory -> `docs/IMPLEMENTATION_INVENTORY_VNEXT_RU.md`.
- Categorical/HoTT appendix -> `docs/CATEGORICAL_HOTT_EXTENSION_RU.md`, `fuzzyxai/category/`, `fuzzyxai/hott/`.
- Category/HoTT report -> `proofs/category_hott_checks.py` -> `reports/category_hott/category_hott_checks.md`.
