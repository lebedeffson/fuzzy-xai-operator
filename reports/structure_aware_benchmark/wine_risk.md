# Structure-aware benchmark: wine_risk

| policy | agreement_reference | missed_critical_ruptures | critical_rupture_recall | false_auto_accept_rate | false_block_rate | auto_accept_coverage |
|---|---:|---:|---:|---:|---:|---:|
| full_observer_calibrated | 0.876984 | 0 | 1.000000 | 0.099206 | 0.000000 | 0.297619 |
| probability_threshold | 0.523810 | 0 | 1.000000 | 0.456349 | 0.000000 | 0.547619 |
| shap_guardrail | 0.523810 | 0 | 1.000000 | 0.456349 | 0.000000 | 0.547619 |
| lime_guardrail | 0.523810 | 0 | 1.000000 | 0.456349 | 0.000000 | 0.547619 |
| anchor_guardrail | 0.523810 | 0 | 1.000000 | 0.456349 | 0.000000 | 0.547619 |

- Structure-aware benchmark uses real rows with controlled explanation-layer perturbations.
- Expected actions are derived from safety rules (rupture/context/trace/reduction constraints).
- agreement_reference is set-based (any safety-compatible action is accepted); agreement_reference_strict uses a single canonical label.