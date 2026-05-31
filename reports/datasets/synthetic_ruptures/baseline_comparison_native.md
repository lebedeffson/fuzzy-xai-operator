# Baseline comparison: synthetic_ruptures

- baseline_access: `native`

| baseline | information_access | agreement_proxy | agreement_reference | missed_critical_ruptures | false_auto_accept_rate | false_block_rate | auto_accept_coverage |
|---|---|---:|---:|---:|---:|---:|---:|
| full_observer_runtime | full_structure | 0.760000 | 0.835556 | 0 | 0.000000 | 0.000000 | 0.151111 |
| full_observer_uncalibrated | full_structure | 0.773333 | 0.822222 | 0 | 0.000000 | 0.000000 | 0.151111 |
| full_observer_calibrated | full_structure | 0.773333 | 0.822222 | 0 | 0.000000 | 0.000000 | 0.151111 |
| probability_threshold | native_risk_only | 0.346667 | 0.253333 | 70 | 0.591111 | 0.000000 | 0.760000 |
| shap_guardrail | native_feature_importance | 0.355556 | 0.253333 | 70 | 0.564444 | 0.000000 | 0.733333 |
| lime_guardrail | native_local_surrogate | 0.346667 | 0.253333 | 70 | 0.586667 | 0.000000 | 0.755556 |
| anchor_guardrail | native_rule_anchor | 0.346667 | 0.253333 | 70 | 0.573333 | 0.000000 | 0.742222 |

- SHAP baseline is included via TreeExplainer.
- LIME baseline is included.
- Anchors baseline is included.
- agreement_proxy is proxy-rule agreement, not clinical expert action accuracy.
- agreement_reference uses independent reference policy (risk + outcome context).