# Воспроизводимые результаты для вставки в статью

Ниже приведён единый набор численных результатов и артефактов, полученных из программной реализации. Все значения взяты из машинно-читаемых отчётов репозитория; `agreement_proxy` означает совпадение с proxy-правилом, а не экспертную клиническую разметку.

## Среда и ExplainPlan

| Параметр | Значение |
| --- | --- |
| Python | `3.14.5` |
| Platform | `Linux-7.0.11-arch1-1-x86_64-with-glibc2.43` |
| ExplainPlan SHA256 | `dc4460c6d7db9df9d3e95dcad6119c99d9c99093911c883d1186573d0cb05473` |
| Chapter 2 YAML ExplainPlan SHA256 | `b3c3fc83fbba3f862e06470ddd1fb9ae77bb57e39c14575534275049e669209e` |
| Mode | `audit` |
| Risk weights | `0.7, 0.05, 0.05, 0.05, 0.15` |
| Thresholds | `0.1, 0.25, 0.5, 0.75` |
| gamma_max / I_min / Delta_max | `0.45 / 0.65 / 0.15` |

## Сквозной операторный пример sample_113

| Поле | Значение |
| --- | --- |
| sample_id | `sample_113` |
| P(malignant) | 0.703577 |
| selected_features | `symmetry error, mean fractal dimension, concave points error, worst fractal dimension, mean texture` |
| mu_low / mu_medium / mu_high | `0.296423 / 0.185692 / 0.703577` |
| active_rules | `['r_high_risk', 'r_low_risk']` |
| U_model / U_rules / U_trace | `0.592846 / 0.296423 / 0.020000` |
| u_M / I_pre | `0.359708 / 0.674324` |
| gamma_model_to_risk | 0.351415 |
| chi_R / chi_R_crit / action | `1 / 1 / block` |

## Редукция и риск-наблюдатель sample_113

| Поле | Значение |
| --- | --- |
| selected_representation | `F_N_src` |
| Delta / I_pre / rho | `0.106811 / 0.674324 / 0.800000` |
| chi_Auto / chi_R / chi_R_crit | `false / 1 / 1` |
| action / reason | `block` / `critical rupture` |

## Breast Cancer Wisconsin: количественная проверка

| Метрика | Значение |
| --- | ---: |
| n | 569 |
| model_accuracy | 0.9720 |
| model_roc_auc | 0.9950 |
| model_f1 / precision / recall | `0.9608 / 1.0000 / 0.9245` |
| agreement_proxy | 0.6084 |
| missed_critical_ruptures | 0 |
| false_auto_accept_rate | 0.0000 |
| mean_I_pre / mean_rho | `0.7936 / 0.2077` |

## Распределения I_pre и rho

| Показатель | mean | std | median | p25 | p75 | p05 | p95 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| I_pre | 0.7936 | 0.0289 | 0.7914 | 0.7811 | 0.8050 | 0.7453 | 0.8422 |
| rho | 0.2077 | 0.1501 | 0.1385 | 0.0613 | 0.3432 | 0.0566 | 0.4606 |

## Калибровка риск-наблюдателя

| Режим | agreement_proxy | agreement_reference | missed_critical | critical_recall | false_auto_accept |
| --- | ---: | ---: | ---: | ---: | ---: |
| before | 0.7063 | 0.7343 | 0 | 1.0000 | 0.0000 |
| after | 0.9580 | 0.9301 | 0 | 1.0000 | 0.0000 |

Калибровочная цель: `maximize agreement_reference - 5*missed_critical_ruptures - 2*false_auto_accept_rate - false_block_rate with hard safety penalties`.
Лучшие параметры: weights=`{'predicted_risk': 0.5, 'uncertainty': 0.2, 'interpretability_gap': 0.1, 'reduction_loss': 0.1, 'diagnostic': 0.1}`, thresholds=`[0.12, 0.28, 0.52, 0.8]`, gamma_max=`0.4`, I_min=`0.6`, Delta_max=`0.12`.

## Synthetic ruptures: native vs equal-guardrail

В режиме `native` baseline-методы получают только собственные входы и не получают `chi_R_crit`; полный наблюдатель получает полный структурный вход.

| Метод | access | agreement_ref | missed_critical | critical_recall | false_auto_accept | auto_accept_cov |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| full_observer_runtime | full_structure | 0.8356 | 0 | 1.0000 | 0.0000 | 0.1511 |
| full_observer_uncalibrated | full_structure | 0.8222 | 0 | 1.0000 | 0.0000 | 0.1511 |
| full_observer_calibrated | full_structure | 0.8222 | 0 | 1.0000 | 0.0000 | 0.1511 |
| probability_threshold | native_risk_only | 0.2533 | 70 | 0.0000 | 0.5911 | 0.7600 |
| shap_guardrail | native_feature_importance | 0.2533 | 70 | 0.0000 | 0.5644 | 0.7333 |
| lime_guardrail | native_local_surrogate | 0.2533 | 70 | 0.0000 | 0.5867 | 0.7556 |
| anchor_guardrail | native_rule_anchor | 0.2533 | 70 | 0.0000 | 0.5733 | 0.7422 |

В режиме `equal_guardrail` всем методам передаётся внешний safety-флаг; это sanity-check политики блокировки.

| Метод | access | agreement_ref | missed_critical | critical_recall | false_auto_accept |
| --- | --- | ---: | ---: | ---: | ---: |
| full_observer_runtime | full_structure | 0.8356 | 0 | 1.0000 | 0.0000 |
| full_observer_uncalibrated | full_structure | 0.8222 | 0 | 1.0000 | 0.0000 |
| full_observer_calibrated | full_structure | 0.8222 | 0 | 1.0000 | 0.0000 |
| probability_threshold | equal_guardrail | 0.8711 | 0 | 1.0000 | 0.0000 |
| shap_guardrail | equal_guardrail | 0.8578 | 0 | 1.0000 | 0.0000 |
| lime_guardrail | equal_guardrail | 0.8667 | 0 | 1.0000 | 0.0000 |
| anchor_guardrail | equal_guardrail | 0.8533 | 0 | 1.0000 | 0.0000 |

## Structure-aware benchmark

Сценарии: `clean, context_forbidden, critical_rupture, high_reduction_loss, rule_conflict, source_conflict, trace_gap`; число кейсов: `252`.

| Policy | agreement_ref | missed_critical | critical_recall | false_auto_accept | auto_accept_cov |
| --- | ---: | ---: | ---: | ---: | ---: |
| full_observer_calibrated | 0.8889 | 0 | 1.0000 | 0.0913 | 0.2698 |
| probability_threshold | 0.5437 | 0 | 1.0000 | 0.4365 | 0.5238 |
| shap_guardrail | 0.5437 | 0 | 1.0000 | 0.4365 | 0.5238 |
| lime_guardrail | 0.5437 | 0 | 1.0000 | 0.4365 | 0.5238 |
| anchor_guardrail | 0.5437 | 0 | 1.0000 | 0.4365 | 0.5238 |

## Улучшения не только на синтетике (real rows + structure-aware)

| Dataset | full_agreement_ref | threshold_agreement_ref | agreement_gain | full_false_auto_accept | threshold_false_auto_accept | false_auto_accept_drop |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| breast_cancer | 0.8889 | 0.5437 | 0.3452 | 0.0913 | 0.4365 | 0.3452 |
| wine_risk | 0.8770 | 0.5238 | 0.3532 | 0.0992 | 0.4563 | 0.3571 |
| diabetes_binary | 0.8810 | 0.8333 | 0.0476 | 0.0238 | 0.0794 | 0.0556 |

## Абляционный анализ

| Mode | agreement_proxy | missed_critical | critical_recall | false_auto_accept | auto_accept_cov | mean_reduction_loss |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| full | 0.6503 | 0 | 1.0000 | 0.0000 | 0.5175 | 0.0788 |
| no_trace | 0.6503 | 0 | 1.0000 | 0.0000 | 0.5175 | 0.0788 |
| no_delta | 0.6573 | 0 | 1.0000 | 0.0000 | 0.5175 | 0.0000 |
| no_critical_rupture | 0.6503 | 0 | 1.0000 | 0.0000 | 0.5175 | 0.0788 |
| f0_only | 0.6294 | 0 | 1.0000 | 0.0000 | 0.5175 | 0.2500 |
| no_topos | 0.6014 | 0 | 1.0000 | 0.0490 | 0.5664 | 0.0788 |
| probability_threshold | 0.8042 | 0 | 1.0000 | 0.1958 | 0.7133 | 0.0788 |

## Команды воспроизведения

- `make dissertation-check`
- `make reproduce-chapter2`
- `make calibrate-chapter2`
- `make benchmark-equal-raw-structure`
- `make chapter3-artifacts`
- `make ecosystem-evidence`
- `make validate-ecosystem-sdk`
- `make thesis-practice-tables`
- `make browser-visual-check`
- `make ui-health-check`
- `make structure-aware-benchmark DATASET=wine_risk`
- `make structure-aware-benchmark DATASET=diabetes_binary`
- `make reproducibility-artifacts`

## Артефакты и SHA256

| Артефакт | Глава | SHA256 | Что подтверждает |
| --- | --- | --- | --- |
| `configs/explain_plan_chapter2.yaml` | 2 / appendix | `4469dad3fbeeb4f53ad953c24bd4e29a8de0d17d6465fb2c71b2b15a954cacfc` | Fixed chapter 2 ExplainPlan YAML contract used for deterministic hashing |
| `reports/chapter2/explain_plan_hash.json` | 2 / appendix | `c18bfa69005fd3b56cc3ea73ee2a59596b00486dd2cdd71398dc5eb30b12e7cb` | Validated ExplainPlan SHA256 and required trace field list |
| `reports/chapter2/sample_113_report.json` | 2 | `4920325789b0e09fc54a9f64f9dd93f0ae5efa82aa57ca74e17903fd97025f33` | Canonical sample_113 report generated from the chapter 2 ExplainPlan contract |
| `reports/chapter2/calibration_constants.csv` | 2 | `5ae7af763efb982ca16108014e443c649a9bca3363f0b7a4e8c8142b88f150ce` | Chapter 2 calibrated c_H/c_O/c_K constants on calibration pairs |
| `reports/chapter2/equal_raw_structure_summary.csv` | 2 | `768da4928dc99f3cdf265f67c8443f30724a2d9f7ef428137148fc2af292706b` | Equal raw structure benchmark for certified route vs raw structural access |
| `reports/chapter3/dataset_roles_summary.csv` | 3 | `e677dd7743b092631a81420c860f9c8e2b45d128cb8f86bda6ad9b3b3f4ea4e2` | Chapter 3 concise dataset role table |
| `api/openapi.yaml` | 4 | `35edef0eea1a6dbf18c30727dfe3282e62bc9dd764812c9e5a913e19c8c23cc3` | Open API contract for /v1/explain and /v1/risk-action |
| `deploy/docker-compose.yml` | 4 / appendix | `b67bcadfd5df765658633f332afeaa53752660281b4ffb149e4f2d87efd24273` | Deployment skeleton for API and Studio services |
| `templates/module_registry_entry.json` | 4 / appendix | `0479e2efe8c63397a00e14068617dba5e463dc3289e5116eeedb83cc3f917f78` | External module registration template |
| `fuzzyxai/ecosystem/registry.json` | 4 | `e794e77071a318559c5b7d2d60cf9f7d4c20e31fb7021f2c15488dd2d47b5df9` | External module registry for chapter 4 ecosystem integration |
| `evidence/evidence_matrix.csv` | 4 / 5 | `97a0eda4f67f2e55925312064d3f4a348c28d9845001110e04ed3b6936805162` | Module evidence matrix with status, fixture, and claim-scope flags |
| `reports/chapter4/ecosystem_evidence.json` | 4 | `d62e72a2d297c817c4039e12e88c7ce3dfad90ae5a0b528870ac7dfc81ce382c` | Chapter 4 external module evidence summary |
| `reports/reproducibility_artifacts/explain_plan.json` | 2 / appendix | `d7cdb8f76152c43aec44316fad634fc337273206b26db67ad890697ab38bf524` | Serializable ExplainPlan contract and trace hash source |
| `reports/chapter2_real_operator_case/breast_cancer_operator_case.json` | 2 | `0d319ff99567c4c8b16969fe22085b6dec1114932875e784f69168b57bd2921d` | sample_113 operator values: mu, alpha, U_model, U_rules, U_trace, u_M, tau |
| `reports/real_reduction_example/breast_cancer_case.json` | 3 | `ca31cd57c9b9d0ddc683a9d42e2deeee0a316a0f2881af408dbe633bdeeca075` | sample_113 reduction, chi_Auto, chi_R, chi_R_crit, rho, action |
| `reports/datasets/breast_cancer/summary.json` | 5 | `e938831464094240f0a88389f421fb8b6a3d58a04e237e87758b466dbb4c3a7e` | Breast Cancer model metrics, observer metrics, I_pre/rho quantiles |
| `reports/datasets/breast_cancer/calibration.json` | 5 | `68118e91c83b2c8f24e5c940bb54dd4adb21620e2734f5eb82ad10ad3dfffc44` | Observer calibration before/after and constrained parameters |
| `reports/datasets/synthetic_ruptures/baseline_comparison_native.json` | 5 | `fad3e831c07ddb943c21a0069b19b40c12bd0f069049377292621efb6d44df6b` | Native-access safety comparison; baselines do not receive chi_R_crit |
| `reports/datasets/synthetic_ruptures/baseline_comparison_equal_guardrail.json` | 5 | `5d7ddd1a679e4aeefc8a9f7f728492b68d11eaac550025d21ec3ac4654782f07` | Equal-guardrail sanity check where all methods receive chi_R_crit |
| `reports/structure_aware_benchmark/breast_cancer.json` | 5 | `dee3b7e6a34a4ae0e11e076d55a39c72d247b565237b08eeda832f255b9827d9` | Structure-aware cases: trace gap, source conflict, context forbidden, high Delta |
| `reports/structure_aware_benchmark/wine_risk.json` | 5 | `84c321f3a1602b87d7756a67b68bf5b986fa399338f1ba2182ab861e5b759121` | Structure-aware benchmark on real wine rows (non-synthetic improvement check) |
| `reports/structure_aware_benchmark/diabetes_binary.json` | 5 | `c8b25086a26888c8d2b44832ee01f7fcb94a1f5d1935977337f76c033467951a` | Structure-aware benchmark on real diabetes rows (non-synthetic improvement check) |
| `reports/datasets/breast_cancer/ablation.json` | 5 | `3a4d678e2890901856badd94348c5dce27cf137231a2d4b68fa304720cba0b54` | Ablation of trace, Delta, chi_R_crit, hierarchy, topos, risk-only threshold |
| `reports/thesis_practice/thesis_practice_tables.json` | appendix | `fc46650590a0fa891bedae193c34606d6489d6e0fce3d5d9c1d8f4534ebcb5c0` | Word/LaTeX-ready thesis practice table index |
| `Makefile` | appendix | `3ee8c7d4d8c7896b4e10e00db0ec4f4f437fa43323e9b55eec8568aa791c92b3` | Reproducible command route |
| `requirements.txt` | appendix | `3202b5309cef65ab6e2ab21ea0f3a659820652d1d944e94dac3df63682a51c57` | Python environment dependencies |

## Корректная формулировка для текста

Встроенные наборы данных используются для количественной проверки и калибровки наблюдающего контура; внешние registry-режимы подтверждают переносимость пайплайна; диагностический режим `synthetic_ruptures` проверяет safety-свойство: `chi_R^crit=1` приводит к `block`, а в native-режиме baseline-методы без доступа к структурному индикатору пропускают критические разрывы.
