# Structure-aware benchmark: diabetes_binary

| policy | agreement_reference | missed_critical_ruptures | critical_rupture_recall | false_auto_accept_rate | false_block_rate | auto_accept_coverage |
|---|---:|---:|---:|---:|---:|---:|
| full_observer_calibrated | 0.634921 | 0 | 1.000000 | 0.047619 | 0.000000 | 0.071429 |
| probability_threshold | 0.500000 | 0 | 1.000000 | 0.079365 | 0.000000 | 0.095238 |
| shap_guardrail | 0.500000 | 0 | 1.000000 | 0.079365 | 0.000000 | 0.095238 |
| lime_guardrail | 0.500000 | 0 | 1.000000 | 0.079365 | 0.000000 | 0.095238 |
| anchor_guardrail | 0.500000 | 0 | 1.000000 | 0.079365 | 0.000000 | 0.095238 |

- Structure-aware benchmark uses real rows with controlled explanation-layer perturbations.
- Expected actions are derived from safety rules (rupture/context/trace/reduction constraints).