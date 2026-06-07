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


## Артефакты и SHA256

| Артефакт | Глава | SHA256 | Что подтверждает |
| --- | --- | --- | --- |

## Корректная формулировка для текста

Встроенные наборы данных используются для количественной проверки и калибровки наблюдающего контура; внешние registry-режимы подтверждают переносимость пайплайна; диагностический режим `synthetic_ruptures` проверяет safety-свойство: `chi_R^crit=1` приводит к `block`, а в native-режиме baseline-методы без доступа к структурному индикатору пропускают критические разрывы.
