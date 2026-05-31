# Baseline comparison: wine_risk

| baseline | agreement_proxy | agreement_reference | missed_critical_ruptures | false_auto_accept_rate | false_block_rate | auto_accept_coverage |
|---|---:|---:|---:|---:|---:|---:|
| full_observer_runtime | 0.622222 | 0.688889 | 0 | 0.000000 | 0.000000 | 0.622222 |
| full_observer_uncalibrated | 0.666667 | 0.733333 | 0 | 0.000000 | 0.000000 | 0.622222 |
| full_observer_calibrated | 0.666667 | 0.733333 | 0 | 0.000000 | 0.000000 | 0.622222 |
| probability_threshold | 1.000000 | 0.933333 | 0 | 0.000000 | 0.000000 | 0.622222 |
| shap_guardrail | 0.977778 | 0.911111 | 0 | 0.000000 | 0.000000 | 0.622222 |
| lime_guardrail | 1.000000 | 0.933333 | 0 | 0.000000 | 0.000000 | 0.622222 |
| anchor_guardrail | 0.977778 | 0.911111 | 0 | 0.000000 | 0.000000 | 0.622222 |

- SHAP baseline is included via TreeExplainer.
- LIME baseline is included.
- Anchors baseline is included.
- agreement_proxy is proxy-rule agreement, not clinical expert action accuracy.
- agreement_reference uses independent reference policy (risk + outcome context).