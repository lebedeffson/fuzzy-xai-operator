# Обновлённые таблицы главы 3

## Таблица A. Natural flow

| n_cases | n_unique_objects | n_conflicts | share_conflicts | n_critical_conflicts | share_critical_conflicts | n_action_diff_f0_nas | share_action_diff_f0_nas | n_f0_accept_nas_block | share_f0_accept_nas_block | n_f0_accept_nas_defer | share_f0_accept_nas_defer | recall_critical_nas | false_auto_accept_f0 | false_auto_accept_nas | false_block_f0 | false_block_nas |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 228 | 228 | 215 | 0.942982 | 209 | 0.916667 | 167 | 0.732456 | 142 | 0.622807 | 0 | 0 | 1 | 142 | 0 | 0 | 0 |

## Таблица B. Conflict-enriched flow

| n_cases | n_unique_objects | n_conflicts | share_conflicts | n_critical_conflicts | share_critical_conflicts | n_action_diff_f0_nas | share_action_diff_f0_nas | n_f0_accept_nas_block | share_f0_accept_nas_block | n_f0_accept_nas_defer | share_f0_accept_nas_defer | recall_critical_nas | false_auto_accept_f0 | false_auto_accept_nas | false_block_f0 | false_block_nas |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1000 | 227 | 938 | 0.938 | 917 | 0.917 | 716 | 0.716 | 605 | 0.605 | 0 | 0 | 1 | 605 | 0 | 0 | 0 |

Object-level CI:

| metric | estimate | case_ci_low | case_ci_high | object_ci_low | object_ci_high | n_cases | n_unique_objects |
| --- | --- | --- | --- | --- | --- | --- | --- |
| share_action_diff_f0_nas | 0.716 | 0.688 | 0.744 | 0.648232 | 0.780266 | 1000 | 227 |
| share_f0_accept_nas_block | 0.605 | 0.574 | 0.636 | 0.533332 | 0.674355 | 1000 | 227 |
| recall_critical_nas | 1 | 1 | 1 | 1 | 1 | 1000 | 227 |
| false_auto_accept_f0 | 0.605 | 0.574 | 0.636 | 0.533332 | 0.674355 | 1000 | 227 |
| false_auto_accept_nas | 0 | 0 | 0 | 0 | 0 | 1000 | 227 |
| false_block_f0 | 0 | 0 | 0 | 0 | 0 | 1000 | 227 |
| false_block_nas | 0 | 0 | 0 | 0 | 0 | 1000 | 227 |

## Таблица C. Baseline comparison

| method | recall_critical | false_auto_accept | unsafe_accept_with_conflict | false_block | objective_J |
| --- | --- | --- | --- | --- | --- |
| B0_probability_only | 0.309706 | 605 | 605 | 0 | 6190 |
| B1_F0_basic | 0.309706 | 605 | 605 | 0 | 6190 |
| B2_F0_plus_uncertainty | 0 | 520 | 520 | 0 | 7185 |
| B3_SHAP_threshold | 0.272628 | 58 | 58 | 0 | 3625 |
| B4_LIME_threshold | 0.0218103 | 208 | 208 | 0 | 5525 |
| B5_rule_only | 0.321701 | 622 | 622 | 0 | 6220 |
| B6_equal_raw_structure_without_NAS | 0 | 24 | 24 | 0 | 4705 |
| M1_NAS | 1 | 0 | 0 | 0 | 0 |
| M2_F_ML | 1 | 0 | 0 | 0 | 0 |

## Таблица D. Calibration v2

| Конфигурация | w_p | w_u | w_I | w_Delta | w_R | theta_1 | theta_2 | theta_3 | theta_4 | J validation | J test | reason selected |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| best_config | 0.00707697 | 0.0180113 | 0.0133202 | 0.943035 | 0.0185568 | 0.163249 | 0.209762 | 0.652227 | 0.870944 | 0 | 0 | manual_config also optimal; selected config has zero proxy loss and tie-breaker-compatible simplicity |

## Таблица E. Ablation

| mode | recall_critical | false_auto_accept | objective_J | effect_vs_full_NAS |
| --- | --- | --- | --- | --- |
| full_NAS | 1 | 0 | 0 | 0 |
| without_rank_conflict | 1 | 0 | 0 | 0 |
| without_sign_conflict | 1 | 0 | 0 | 0.013 |
| without_rule_conflict | 1 | 0 | 0 | 0 |
| without_source_labels | 1 | 0 | 0 | 0.013 |
| without_delta_reduction | 1 | 0 | 0 | 0 |
| without_chi_R_crit | 0 | 518 | 7175 | 0.917 |
| F0_only | 0.309706 | 605 | 6190 | 0.716 |