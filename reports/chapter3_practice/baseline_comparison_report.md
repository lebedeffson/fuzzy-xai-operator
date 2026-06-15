# Baseline comparison

Сравнение `B6_equal_raw_structure_without_NAS` отделяет доступ к сырым признакам от наличия источникового представления. Поэтому результат NAS/F_ML интерпретируется как эффект структурного сохранения источника конфликта, а не как эффект передачи дополнительной скрытой метки.

| method | recall_critical | missed_critical | false_auto_accept | unsafe_accept_with_conflict | false_block | excessive_defer | action_diff_vs_F0 | objective_J |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| B0_probability_only | 0.309706 | 633 | 605 | 605 | 0 | 0 | 0 | 6190 |
| B1_F0_basic | 0.309706 | 633 | 605 | 605 | 0 | 0 | 0 | 6190 |
| B2_F0_plus_uncertainty | 0 | 917 | 520 | 520 | 0 | 0 | 0.457 | 7185 |
| B3_SHAP_threshold | 0.272628 | 667 | 58 | 58 | 0 | 0 | 0.607 | 3625 |
| B4_LIME_threshold | 0.0218103 | 897 | 208 | 208 | 0 | 0 | 0.686 | 5525 |
| B5_rule_only | 0.321701 | 622 | 622 | 622 | 0 | 0 | 0.041 | 6220 |
| B6_equal_raw_structure_without_NAS | 0 | 917 | 24 | 24 | 0 | 0 | 0.958 | 4705 |
| M1_NAS | 1 | 0 | 0 | 0 | 0 | 0 | 0.716 | 0 |
| M2_F_ML | 1 | 0 | 0 | 0 | 0 | 0 | 0.716 | 0 |