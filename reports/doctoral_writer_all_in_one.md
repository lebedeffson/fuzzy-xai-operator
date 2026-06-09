# Единый пакет для писателя по главам 2-5

Этот файл можно отдавать писателю как единую справку: что реализовано, какие числа вставлять, где артефакты и какие утверждения запрещены.

## 1. Статус репозитория

- Ветка: `feature/math-aligned-code`
- HEAD: `7509c81`
- Последние коммиты:

```text
7509c81 Add chapter 2-3 final evidence matrix
00eb639 Update writer handoff for Beacon and GIS
5cf39c8 Add Beacon and GIS executable scenario adapters
16c3879 Add doctoral writer handoff brief
a12a472 Close doctoral evidence integration gaps
c903f2f Add chapter 3 controlled critical ruptures experiment
a94cb63 Add external scenario baseline comparison tables
8472b52 Clarify chapter 5 scenario claim scope
```

## 2. Проверки

- `pytest -q`: `210 passed`, предупреждения только от `sklearn` про feature names.
- `make doctoral-final-evidence`: `PASS`.
- `make chapter2-3-final-evidence`: `PASS`.
- Матрица глав 2-3: `15/15` реализовано, `missing = []`.
- Финальный manifest: `ok`.

Ключевые файлы проверки:

- `reports/doctoral_final_evidence_index.md`
- `evidence/doctoral_final_manifest_sha256.json`
- `reports/chapter2_3/chapter2_3_final_evidence.json`
- `evidence/chapter2_3_manifest_sha256.json`

## 3. Главный принцип для текста

Сценарии без закрепленного источника или без pinned baseline нельзя использовать как количественное экспериментальное доказательство. Для них допустимы только формулировки вида `adapter/report route`, `fixture-сценарий`, `source-pending`, `planned`, `audit_report`, `not_run`.

Нельзя придумывать `accuracy`, `missed critical ruptures`, `false auto accept` для внешних модулей, если в таблицах стоит `N/A` или `not_available`.

## 4. Глава 2: что реализовано

Глава 2 закрыта не текстовой правкой, а программной матрицей: каждый объект имеет код, тест и артефакт.

| Объект | Что закрывает | Статус | Код | Тесты | Артефакты |
| --- | --- | --- | --- | --- | --- |
| ExplainPlan | Контракт ExplainPlan: термы, правила, веса, пороги и trace | implemented | fuzzyxai/core/explain_plan.py; configs/explain_plan_chapter2.yaml | tests/test_explain_plan_contract.py | reports/chapter2/explain_plan_hash.json |
| E_k | Объект объяснения ExplanationObject E_k | implemented | fuzzyxai/core/explanation_object.py; fuzzyxai/core/system_operator.py | tests/test_operator_trace.py | reports/chapter2/sample_113_report.json; reports/chapter2/sample_113_trace.json |
| Omega | Системный оператор для скалярного риска | implemented | fuzzyxai/core/system_operator.py; fuzzyxai/experiments/chapter2_sample113.py | tests/test_operator_trace.py; tests/test_chapter2_breast_cancer_demo_smoke.py | reports/chapter2/sample_113_table.csv |
| d_E_gamma_Dij | Семантическое рассогласование d_E, gamma и D_ij | implemented | fuzzyxai/core/trust_evaluator.py; fuzzyxai/core/composition.py; fuzzyxai/core/diagnostics.py | tests/test_semantic_disagreement_pseudometric.py; tests/test_composition.py; tests/test_composition_well_defined.py | reports/chapter2/equal_raw_structure_report.json |
| T_ij_synthesis | Ограниченный синтез перехода T_ij | implemented | fuzzyxai/core/alignment_synthesis.py; experiments/chapter2_alignment_synthesis.py | tests/test_alignment_synthesis.py | reports/chapter2/alignment_synthesis_report.json |
| I_EG_L_E | Потеря интерпретируемости L(E) и индекс I(E_G) | implemented | fuzzyxai/core/trust_evaluator.py | tests/test_trust.py; tests/test_chain_loss_bound.py | reports/chapter2/chapter2_breast_cancer_summary.json |
| cH_cO_cK_calibration | Калибровка констант c_H, c_O, c_K | implemented | fuzzyxai/experiments/chapter2_calibration.py | tests/test_chapter2_chapter3_artifacts.py | reports/chapter2/calibration_report.json; reports/chapter2/calibration_constants.csv; figures/chapter2/calibration_constants.png |
| equal_raw_structure | Benchmark native/equal_guardrail/equal_raw_structure | implemented | fuzzyxai/experiments/chapter2_equal_raw_structure.py | tests/test_chapter2_chapter3_artifacts.py | reports/chapter2/equal_raw_structure_summary.csv; figures/chapter2/equal_raw_structure_comparison.png |

### Готовая формулировка для главы 2

> Для главы 2 сформирована воспроизводимая программная матрица: `ExplainPlan`, объект объяснения `E_k`, системный оператор, композиция, `d_E/gamma/D_ij`, ограниченный синтез `T_ij`, индексы `L(E)/I(E_G)`, калибровка `c_H/c_O/c_K` и benchmark `equal_raw_structure` сопоставлены с исходным кодом, pytest-проверками и машинно-читаемыми артефактами. Команда `make chapter2-3-final-evidence` завершается успешно только при наличии всех подтверждений.

### Ограниченный синтез T_ij

- Код: `fuzzyxai/core/alignment_synthesis.py`
- Отчёт: `reports/chapter2/alignment_synthesis_report.json`
- Результат: `gamma=0.02`, `Delta_T=0.0`, `J(T)=0.02`.

Корректная формулировка:

> Синтез перехода `T_ij` реализован только в ограниченном виде: кандидатные преобразования перечисляются конечным образом в `ExplainPlan`. Если пространство кандидатов не задано или минимальное значение `J(T)` превышает `gamma_max`, система возвращает `DiagnosticState`, а не строит произвольное преобразование.

## 5. Глава 3: что реализовано

| Объект | Что закрывает | Статус | Код | Тесты | Артефакты |
| --- | --- | --- | --- | --- | --- |
| F_hierarchy | Иерархия F0, IntervalFS, HesitantFS, NeutrosophicFS, MultiLevelFS | implemented | fuzzyxai/hierarchy/f0.py; fuzzyxai/hierarchy/interval.py; fuzzyxai/hierarchy/hesitant.py; fuzzyxai/hierarchy/neutrosophic.py; fuzzyxai/hierarchy/multilevel.py | tests/test_reductions.py; tests/test_multilevel_reduction_termination.py | dissertation_artifacts/diagram_specs/chapter3/fig_3_2_hierarchy.json |
| Reduction_Delta | Редукция к F0 и потеря редукции Delta | implemented | fuzzyxai/hierarchy/reductions.py; fuzzyxai/hierarchy/meta_reducer.py | tests/test_reduction_approximation_theorem.py; tests/test_reduction_graph.py | dissertation_artifacts/diagram_specs/chapter3/fig_3_3_reduction.json |
| P_sit_D_choice | Профиль ситуации P_sit, Pareto-выбор и D_choice | implemented | fuzzyxai/selection/profile_builder.py; fuzzyxai/selection/pareto_selector.py; fuzzyxai/selection/choice_diagnostic.py | tests/test_selector.py; tests/test_operational_coverage_minimality.py | reports/chapter3/dataset_roles_summary.csv |
| chi_Auto_topos | Контекстная допустимость chi_Auto и topoi/subpresheaf слой | implemented | fuzzyxai/category/context_topos.py; fuzzyxai/category/subpresheaf.py; fuzzyxai/risk/context_acceptance.py | tests/test_characteristic_morphism_auto.py; tests/test_chi_auto_truth_values.py; tests/test_subobject_classifier.py; tests/test_subpresheaf.py | dissertation_artifacts/diagram_specs/chapter3/fig_3_8_chi_auto_sample113.json |
| CertifiedPath_Rupture | CertifiedPath и Rupture объяснительного маршрута | implemented | fuzzyxai/category/certified_path.py; fuzzyxai/hott/rupture_type.py; fuzzyxai/hott/path_certificates.py | tests/test_certified_path_category.py; tests/test_hott_certified_paths.py; tests/test_hott_path_rupture.py | dissertation_artifacts/diagram_specs/chapter3/fig_3_4_route.json |
| rho_action | rho(x), policy и действия наблюдателя | implemented | fuzzyxai/risk/risk_function.py; fuzzyxai/risk/decision_policy.py; fuzzyxai/risk/observer_pipeline.py | tests/test_risk_function.py; tests/test_decision_policy.py; tests/test_critical_rupture_blocks.py | reports/chapter3/synthetic_ruptures_summary.json |
| controlled_critical_ruptures | Controlled critical ruptures: пять типов нарушений и пять политик | implemented | fuzzyxai/experiments/chapter3_critical_ruptures.py | tests/test_chapter3_critical_ruptures.py | reports/chapter3/synthetic_ruptures_results.csv; figures/chapter3/critical_ruptures_comparison.png; evidence/critical_ruptures_manifest_sha256.json |

### Controlled critical ruptures

- Объектов: `1000`.
- Seed: `314159`.
- Типы нарушений: `trace_gap, rule_conflict, term_inversion, source_conflict, missing_certified_transition`.
- Команда: `make reproduce-critical-ruptures`.
- ExplainPlan SHA256: `4469dad3fbeeb4f53ad953c24bd4e29a8de0d17d6465fb2c71b2b15a954cacfc`.
- Input checksum: `b781681c74978f566a252e993d728d28455138c25b53821c03a833c24f5f4a9a`.
- Results checksum: `494c6cac3a3a94c1b91db074e4d3e4106df9aad7f1bc15f8cf18a7c51ffcac0e`.

| policy | n | critical | missed | recall | false block | agreement | ms/object |
| --- | --- | --- | --- | --- | --- | --- | --- |
| probability_threshold | 1000 | 250 | 250 | 0.0 | 0.0013333333333333333 | 0.557 | 0.0001 |
| shap_rule | 1000 | 250 | 206 | 0.176 | 0.0 | 0.602 | 0.0001 |
| lime_rule | 1000 | 250 | 196 | 0.216 | 0.0 | 0.612 | 0.0001 |
| anchors_rule | 1000 | 250 | 234 | 0.064 | 0.0 | 0.574 | 0.0001 |
| fuzzyxai_full_contour | 1000 | 250 | 0 | 1.0 | 0.0 | 1.0 | 0.0001 |

Готовая формулировка:

> На диагностическом стенде controlled critical ruptures полный контур FuzzyXAI не пропускает критические разрывы (`missed_critical_ruptures=0`, `critical_rupture_recall=1.0`), тогда как baseline-режимы без полного структурного маршрута пропускают часть критических нарушений.

Ограничение:

> Этот результат относится к контролируемому диагностическому стенду. Он проверяет safety-свойство маршрута, но не является клинической валидацией на реальных медицинских данных.

## 6. Глава 4: экосистема и трудоёмкость

- Evidence-пакет: `evidence/evidence_matrix.csv`, `evidence/registry_snapshot.json`, `evidence/reproduction_index.md`.
- SDK/API/Docker/workflow регистрации есть в репозитории: `fuzzyxai/sdk/`, `api/openapi.yaml`, `deploy/`, `scripts/register_external_module.py`.
- GUI/evidence screenshots и dissertation artifacts лежат в `dissertation_artifacts/` и `reports/browser_visual_check/`.

### Протокол трудоёмкости

- measurement_count: `4`
- total_duration_minutes: `79`
- semantic_gap_detected_count: `4`
- claim_scope: `integration effort protocol only; not production estimate`
- source_csv: `reports/chapter4/integration_effort_measurements.csv`

Готовая формулировка:

> Оценки трудоёмкости приводятся только как протоколированные измерения отдельных интеграционных действий. Они не являются оценкой production-внедрения и не должны обобщаться без дополнительного протокола.

## 7. Глава 5: сценарии внешних модулей

Главная таблица: `reports/chapter5/scenario_run_summary.csv`.

| registry_id | source_repo | adapter | output | E | D | action | status | claim_scope |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| hybrid_xiris | https://github.com/lebedeffson/hybrid-xiris-biometric | True | ExplanationArtifact + report | True | True | audit_report | real-output-compatible | adapter compatibility and evidence routing, not local model retraining |
| gd_anfis_shap | source-not-provided:gd_anfis_shap | True | ExplanationArtifact + report | True | False | audit_report | source-pending | контрольный маршрут (исполняемый артефакт); source-pending; качество исходной модели не заявляется |
| anza_lira | https://github.com/fims9000/anza_lira | True | ExplanationArtifact + report | True | False | audit_report | fixture-certified | vessel-segmentation adapter route and report generation, not local model retraining |
| beacon_xai | https://github.com/fims9000/BeaconXAI | True | ExplanationArtifact + report | True | False | audit_report | fixture-certified | executable BeaconXAI fixture adapter with pinned repo HEAD; no local model retraining or quantitative claim |
| gis_integro | source-not-provided:gis_integro | True | ExplanationArtifact + report | True | False | audit_report | source-pending | контрольный маршрут (исполняемый GIS fixture через GD-ANFIS/SHAP-каналы); source-pending; качество исходной модели не заявляется |
| deep_neuro_fuzzy_kafn | https://github.com/lebedeffson/deep-neuro-fuzzy | True | ExplanationArtifact + report | True | False | audit_report | fixture-certified | fixed fixture adapter run with source provenance |
| fan_multimodal | https://github.com/lebedeffson/FuzzyAttentionNetworks | True | ExplanationArtifact + report | True | False | audit_report | fixture-certified | concept-level fuzzy attention scenario |
| kan_xai_2_system | https://github.com/fims9000/KAN-XAI-2.0-System | True | ExplanationArtifact + report | True | False | audit_report | fixture-certified | medical image explanation adapter route, not local model retraining |
| trust_ade | https://github.com/fims9000/Trust-ADE | True | ExplanationArtifact + report | True | False | audit_report | fixture-certified | trust-assessment report route and observer compatibility |
| synt_isic_timeshap | https://github.com/fims9000/SYNT_ISIC | True | ExplanationArtifact + report | True | False | audit_report | fixture-certified | temporal explanation route and report generation, not local dermatology validation |
| eaar_regularization | https://github.com/lebedeffson/eaar-regularization | False | registered metadata only | False | False | not_run | planned | registered future adapter, excluded from quantitative claims |
| neutron_shap_reconstruction | https://github.com/lebedeffson/bonner-shap-reconstruction | False | registered metadata only | False | False | not_run | planned | registered future adapter, excluded from quantitative claims |

### BEACON-XAI

- Код: `fuzzyxai/adapters/beacon_xai.py`.
- Fixture: `data/article_fixtures/beacon_xai_output.json`.
- Репозиторий: `https://github.com/fims9000/BeaconXAI`.
- Commit: `660366759fb0b5045491a9f7b9fa50745afe44db`.
- Статус: `fixture-certified`.
- Claim: исполняемый adapter/report route, без количественного сравнения качества модели.

### GIS INTEGRO

- Код: `fuzzyxai/adapters/gis_integro.py`.
- Fixture: `data/article_fixtures/gis_integro_output.json`.
- Статус: `source-pending`.
- Каналы: GD-ANFIS/SHAP-style route (`R_k`, `alpha_k`, `eta_k`, `u_k`, `tau_k`).
- Claim: контрольный маршрут (исполняемый GIS fixture), без quantitative claims до закрепления внешнего источника.


### GIS INTEGRO route metrics

- Отчёт: `reports/chapter5/gis_integro_route_metrics.json`.
- `probability = 0.67`.
- `mean_alpha_k = 0.72`.
- `positive_SHAP_support = 0.47`.
- `gamma_route = 0.20`.
- `Delta = 0.08`.

Готовая фраза для раздела 5.6:

> В контрольном прогоне FuzzyXAI для GIS INTEGRO получено `gamma_route = 0.20` и `Delta = 0.08`; величина `gamma_route` вычислена как максимум рассогласования между вероятностью модели, средним уровнем активации `alpha_k` и положительной SHAP-поддержкой. Это число фиксирует маршрутное рассогласование контрольного fixture, но не является сравнением качества исходной GIS-модели.

### GD-ANFIS/SHAP

- Код: `fuzzyxai/adapters/gd_anfis_shap.py`.
- Fixture: `data/article_fixtures/gd_anfis_shap_output.json`.
- Статус: `source-pending`.
- Каналы: правила -> `R_k`, активации -> `alpha_k`, SHAP -> `eta_k`, неопределённость -> `u_k`, запуск/источник -> `tau_k`.

Готовая формулировка:

> BEACON-XAI подключён как исполняемый fixture-маршрут с закреплённым репозиторием и commit hash. GIS INTEGRO и GD-ANFIS/SHAP подключены как исполняемые fixture-маршруты через GD-ANFIS/SHAP-каналы, но сценарии со статусом `source-pending` не используются как количественное экспериментальное подтверждение. Все эти сценарии показывают прохождение внешнего артефакта через адаптер и отчётный контур.

## 8. HYBRID-XIRIS blocking case

- Отчёт: `reports/chapter5/hybrid_xiris_blocking_case.json`.
- CSV: `reports/chapter5/hybrid_xiris_blocking_case.csv`.
- Figure: `figures/chapter5/hybrid_xiris_blocking_case.png`.

| sample_id | image_quality_score | segmentation_quality_score | model_match_score | rule_model_accept_activation | rule_quality_block_activation | source_conflict | chi_R_crit | chi_Auto | rho | action | explain_plan_hash | fixture_sha256 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| hybrid_xiris_block_001 | 0.31 | 0.27 | 0.88 | 0.82 | 0.91 | True | 1 | False | 0.8 | block | 4469dad3fbeeb4f53ad953c24bd4e29a8de0d17d6465fb2c71b2b15a954cacfc | bcc3bfddaaa2b70b1bc6cd86fced7c00572fee57e7377abae4a6a809fb690721 |

Готовая формулировка:

> Контрольный пример HYBRID-XIRIS показывает, что высокий `model_match_score` сам по себе не разрешает автоматическое применение: при низком качестве сегментации и конфликте источников устанавливается `chi_R_crit=1`, `chi_Auto=false`, и действие становится `block`.

## 9. Baseline для внешних сценариев

Таблица показывает, что искусственные accuracy/missed critical ruptures не подставлены. Если pinned baseline отсутствует, стоит `not_available` или `N/A`.

| registry_id | baseline_metric | baseline_value | fuzzyxai_metric | fuzzyxai_value | missed_critical_ruptures | false_auto_accept_rate | quantitative_comparison_available | status | note |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| hybrid_xiris | accuracy | not_available | safe_accuracy | not_available | N/A | N/A | false | real-output-compatible | No fake quantitative value is inserted; source metric must be pinned before comparison. |
| gd_anfis_shap | not_available | not_available | not_available | not_available | N/A | N/A | false | source-pending | No fake quantitative value is inserted; source metric must be pinned before comparison. |
| anza_lira | dice | not_available | safe_dice | not_available | N/A | N/A | false | fixture-certified | No fake quantitative value is inserted; source metric must be pinned before comparison. |
| beacon_xai | not_available | not_available | not_available | not_available | N/A | N/A | false | fixture-certified | No fake quantitative value is inserted; source metric must be pinned before comparison. |
| gis_integro | not_available | not_available | not_available | not_available | N/A | N/A | false | source-pending | No fake quantitative value is inserted; source metric must be pinned before comparison. |
| deep_neuro_fuzzy_kafn | f1 | not_available | safe_f1 | not_available | N/A | N/A | false | fixture-certified | No fake quantitative value is inserted; source metric must be pinned before comparison. |
| fan_multimodal | attention_consistency | not_available | safe_attention_consistency | not_available | N/A | N/A | false | fixture-certified | No fake quantitative value is inserted; source metric must be pinned before comparison. |
| kan_xai_2_system | not_available | not_available | not_available | not_available | N/A | N/A | false | fixture-certified | No fake quantitative value is inserted; source metric must be pinned before comparison. |
| trust_ade | not_available | not_available | not_available | not_available | N/A | N/A | false | fixture-certified | No fake quantitative value is inserted; source metric must be pinned before comparison. |
| synt_isic_timeshap | not_available | not_available | not_available | not_available | N/A | N/A | false | fixture-certified | No fake quantitative value is inserted; source metric must be pinned before comparison. |
| eaar_regularization | not_available | not_available | not_available | not_available | N/A | N/A | false | planned | No fake quantitative value is inserted; source metric must be pinned before comparison. |
| neutron_shap_reconstruction | not_available | not_available | not_available | not_available | N/A | N/A | false | planned | No fake quantitative value is inserted; source metric must be pinned before comparison. |

## 10. Что вставлять напрямую

1. Глава 2: абзац про матрицу реализации и ограниченный синтез `T_ij`.
2. Глава 3: таблицу controlled critical ruptures из раздела 5 этого файла или из `reports/chapter3/synthetic_ruptures_results.csv`.
3. Глава 4: evidence/registry + протокол трудоёмкости, строго с ограничением `not production estimate`.
4. Глава 5: `scenario_run_summary.csv`, описание adapter/report route, HYBRID-XIRIS blocking case.

## 11. Что нельзя утверждать

- Нельзя писать, что внешние сценарии дают количественное превосходство, если в baseline-таблице стоит `not_available`.
- Нельзя писать клиническую валидность по synthetic ruptures.
- Нельзя повышать `source-pending` до `fixture-certified` без закреплённого источника и fixture.
- Нельзя использовать протокол трудоёмкости как production estimate.
- Нельзя писать, что GUI является отдельной теорией: GUI является единой демонстрационной оболочкой для уже реализованных операторов, маршрутов, evidence и отчётов.

## 12. Команды воспроизведения

```bash
make chapter2-3-final-evidence
make reproduce-critical-ruptures
make ecosystem-evidence
make dissertation-artifacts
make doctoral-final-evidence
PYTHONPATH=. /home/lebedeffson/Code/venv/bin/python -m pytest -q
```

## 13. Короткая итоговая формулировка для введения к практической части

> Практическая реализация FuzzyXAI оформлена как единый воспроизводимый контур: математические объекты глав 2-3 сопоставлены с кодом, тестами и артефактами; глава 4 фиксирует открытые интерфейсы, реестр модулей и evidence-пакет; глава 5 демонстрирует прохождение внешних сценариев через adapter/report route без завышения количественных claims. Для safety-проверки используется controlled critical ruptures, где полный контур FuzzyXAI обнаруживает все критические разрывы, а baseline-режимы без полного структурного маршрута часть разрывов пропускают.
