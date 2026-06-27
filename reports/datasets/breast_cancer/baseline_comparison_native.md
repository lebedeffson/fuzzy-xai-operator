# Baseline comparison: breast_cancer

- baseline_access: `native`

| baseline | information_access | agreement_proxy | agreement_reference | missed_critical_ruptures | false_auto_accept_rate | false_block_rate | auto_accept_coverage |
|---|---|---:|---:|---:|---:|---:|---:|
| full_observer_runtime | full_structure | 0.608392 | 0.643357 | 0 | 0.000000 | 0.000000 | 0.559441 |
| full_observer_uncalibrated | full_structure | 0.636364 | 0.671329 | 0 | 0.000000 | 0.000000 | 0.559441 |
| full_observer_calibrated | full_structure | 0.923077 | 0.888112 | 0 | 0.000000 | 0.000000 | 0.552448 |
| probability_threshold | native_risk_only | 0.846154 | 0.811189 | 0 | 0.153846 | 0.000000 | 0.713287 |
| shap_guardrail | native_feature_importance | 0.846154 | 0.811189 | 0 | 0.097902 | 0.000000 | 0.657343 |
| lime_guardrail | native_local_surrogate | 0.846154 | 0.811189 | 0 | 0.153846 | 0.000000 | 0.713287 |
| anchor_guardrail | native_rule_anchor | 0.846154 | 0.811189 | 0 | 0.153846 | 0.000000 | 0.713287 |

- SHAP baseline is included via TreeExplainer.
- LIME baseline is included.
- Anchors baseline is included.
- agreement_proxy is proxy-rule agreement, not clinical expert action accuracy.
- agreement_reference uses independent reference policy (risk + outcome context).
- For breast_cancer, reference is risk-dominated; probability-threshold baseline is expected to be strong.