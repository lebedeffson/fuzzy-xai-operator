# Полные таблицы по компонентам системы FuzzyXAI

## 1) Компоненты системы

| Этап | Объект | Вход | Выход | Метрика | Код | Пояснение |
| --- | --- | --- | --- | --- | --- | --- |
| Dataset | DatasetCase | raw/local dataset mode | normalized case data | status, rows, domain | fuzzyxai/datasets/*, apps/services/layered_case.py | Подготовка данных в единый формат для последующих слоёв. |
| Model | prediction, p(y=1), uncertainty | features(x) | prediction, predicted_risk | accuracy, roc_auc | fuzzyxai/pipelines/dataset_observer_pipeline.py | Базовый прогноз без принятия финального решения. |
| Omega | E_model=<L,mu,R,alpha,u,tau> | model output | structured explanation object | u, active rules | fuzzyxai/core/system_operator.py | Перевод численного прогноза в проверяемую объяснительную структуру. |
| Expl / Rupture | T_ij or D_ij | E_model -> E_risk -> E_action | morphism / rupture | gamma, gamma_max, chi_R | fuzzyxai/core/composition.py, fuzzyxai/core/diagnostics.py | Проверка согласованности переходов между слоями. |
| Fuzzy Representation | F0 / F_int / F_H / F_N_src / FML-audit | uncertainty profile | selected_class, Delta | coverage, complexity, reduction_loss | fuzzyxai/risk/representation_selection.py, apps/services/layered_case.py | Выбор минимально достаточного класса неопределенности. |
| Context Topos | RiskContext, AutoAccept, chi_Auto | context presheaves | auto apply allowed/denied | AutoAccept(E) | fuzzyxai/category/context_topos.py, fuzzyxai/category/subpresheaf.py | Контекстно-зависимый фильтр для безопасного автоприменения. |
| Risk Observer | rho, chi_R, chi_R^crit, action | risk components | accept/lower/request/defer/block | mean_rho, action_distribution | fuzzyxai/risk/*, apps/layered_demo.py | Интегральный риск и пороговая политика безопасного действия. |
| Reports | JSON/CSV/MD | pipeline artifacts | dissertation tables | reproducibility | experiments/dataset_benchmark.py, experiments/dissertation_demo_summary.py | Фиксация воспроизводимых результатов для защиты. |

## 2) Датасеты и готовность

| Dataset mode | Domain | Rows | Status | Pipeline | What validates |
| --- | --- | --- | --- | --- | --- |
| breast_cancer | medical | 569 | READY | True | Medical risk baseline: probability + observer action policy. |
| diabetes_binary | medical | 442 | READY | True | Borderline medical uncertainty: lower_confidence/request_more_data behavior. |
| wine_risk | general | 178 | READY | True | Cross-domain robustness check beyond one medical dataset. |
| synthetic_ruptures | controlled | 900 | READY | True | Controlled rupture generation for Expl/HoTT and safe-action diagnostics. |
| registry_programs | tabular_text | 10007 | READY | True | Text+tabular explainability/ranking scenario from CIT registry. |
| registry_mosmed_doctor_analysis | medical_audit | 10 | READY | True | Medical audit scenario: doctor/model decision alignment. |
| registry_steel_ir | industrial_cv | 8080 | READY | True | Industrial control scenario: transfer beyond medical domain. |

## 3) Количественная проверка (встроенные режимы)

| Dataset | Acc | ROC AUC | Observer action acc | Observer proxy acc | Rupture rate | Crit rupture rate | positive_rate | score_std | Note |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| breast_cancer | 0.972028 | 0.994969 | N/A | 0.559441 | 0 | 0 | 0.372583 | 0.422086 | Prototype measurements per object; no I/O timing. |
| diabetes_binary | 0.756757 | 0.82987 | N/A | 0.099099 | 0 | 0 | 0.5 | 0.274921 | Prototype measurements per object; no I/O timing. Stress-test for borderline uncertainty; threshold calibration may be required. |
| wine_risk | 0.977778 | 1 | N/A | 0.622222 | 0 | 0 | 0.269663 | 0.390654 | Prototype measurements per object; no I/O timing. |
| synthetic_ruptures | 0.951111 | 0.995591 | N/A | 0.226667 | 0.346667 | 0.222222 | 0.442222 | 0.35258 | Prototype measurements per object; no I/O timing. Rupture proxies are derived from expert/source disagreement fields. |

## 4) Registry-режимы: readiness и ограничения интерпретации

| Dataset | Pipeline | Observer action acc applicable | Observer action acc | ROC reason | Limitation |
| --- | --- | --- | --- | --- | --- |
| registry_programs | True | False | N/A | roc_auc near 0.5: ranking signal is weak or class imbalance dominates | Prototype measurements per object; no I/O timing. Registry mode validates readiness/portability of the pipeline; action quality metric may be N/A. No expert action labels: observer_action_accuracy is not applicable. |
| registry_mosmed_doctor_analysis | True | False | N/A | roc_auc undefined: single class in test split | Prototype measurements per object; no I/O timing. Registry mode validates readiness/portability of the pipeline; action quality metric may be N/A. Small audit slice (n=10): not for statistical validation. |
| registry_steel_ir | True | False | N/A | roc_auc near 0.5: ranking signal is weak or class imbalance dominates | Prototype measurements per object; no I/O timing. Registry mode validates readiness/portability of the pipeline; action quality metric may be N/A. ROC AUC must not be interpreted as quality here; use this mode for industrial contour portability. |

## 5) Кейсы наблюдателя по слоям

| Scenario | predicted_risk | uncertainty | selected_class | Delta | I_pre | rho | chi_R | chi_R_crit | action |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| safe | 0.000531 | 0.001061 | FML-audit | 0.07 | 0.901666 | 0.005341 | 0 | 0 | accept |
| ambiguous | 0.514468 | 0.971064 | FML-audit | 0.07 | 0.610124 | 0.8 | 1 | 1 | block |
| rupture | 0.514468 | 0.971064 | FML-audit | 0.07 | 0.610124 | 0.8 | 1 | 1 | block |

## 6) Пороговая политика

| Условие | Действие | Пояснение |
| --- | --- | --- |
| chi_R^crit(x)=1 | block | критический разрыв |
| rho(x)>=0.80 and chi_R^crit(x)=0 | defer_to_human | высокий риск без крит. разрыва |
| rho(x)<0.80 and chi_R(x)=1 and chi_R^crit(x)=0 | request_more_data | некритический разрыв |

Примечание: для MosMed оригинальный архив ~104.9GB, в локальном контуре используется малый табличный аудит-слепок.