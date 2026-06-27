# Ablation study

Удаление source labels снижает способность отличать поддержку от контрсвидетельства. Удаление chi_R_crit превращает NAS в обычный числовой риск и повышает unsafe accept.

| mode | recall_critical | missed_critical | false_auto_accept | unsafe_accept_with_conflict | false_block | excessive_defer | action_diff_vs_F0 | objective_J | effect_vs_full_NAS |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| full_NAS | 1 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| without_rank_conflict | 1 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| without_sign_conflict | 1 | 0 | 0 | 0 | 0 | 0 | 0.013 | 0 | 0.013 |
| without_rule_conflict | 1 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| without_source_labels | 1 | 0 | 0 | 0 | 0 | 0 | 0.013 | 0 | 0.013 |
| without_delta_reduction | 1 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| without_chi_R_crit | 0 | 917 | 518 | 518 | 0 | 0 | 0.917 | 7175 | 0.917 |
| F0_only | 0.309706 | 633 | 605 | 605 | 0 | 0 | 0.716 | 6190 | 0.716 |