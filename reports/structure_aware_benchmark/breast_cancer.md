# Structure-aware benchmark: breast_cancer

| policy | agreement_reference | missed_critical_ruptures | critical_rupture_recall | false_auto_accept_rate | false_block_rate | auto_accept_coverage |
|---|---:|---:|---:|---:|---:|---:|
| full_observer_calibrated | 0.888889 | 0 | 1.000000 | 0.091270 | 0.000000 | 0.269841 |
| probability_threshold | 0.543651 | 0 | 1.000000 | 0.436508 | 0.000000 | 0.523810 |
| shap_guardrail | 0.543651 | 0 | 1.000000 | 0.436508 | 0.000000 | 0.523810 |
| lime_guardrail | 0.543651 | 0 | 1.000000 | 0.436508 | 0.000000 | 0.523810 |
| anchor_guardrail | 0.543651 | 0 | 1.000000 | 0.436508 | 0.000000 | 0.523810 |

- Structure-aware benchmark uses real rows with controlled explanation-layer perturbations.
- Expected actions are derived from safety rules (rupture/context/trace/reduction constraints).
- agreement_reference is set-based (any safety-compatible action is accepted); agreement_reference_strict uses a single canonical label.