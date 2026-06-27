# Baseline comparison: diabetes_binary

| baseline | agreement_proxy | agreement_reference | missed_critical_ruptures | false_auto_accept_rate | false_block_rate | auto_accept_coverage |
|---|---:|---:|---:|---:|---:|---:|
| full_observer_runtime | 0.675676 | 0.720721 | 0 | 0.000000 | 0.000000 | 0.099099 |
| full_observer_uncalibrated | 0.684685 | 0.729730 | 0 | 0.000000 | 0.000000 | 0.099099 |
| full_observer_calibrated | 0.684685 | 0.729730 | 0 | 0.000000 | 0.000000 | 0.099099 |
| probability_threshold | 1.000000 | 0.846847 | 0 | 0.000000 | 0.000000 | 0.099099 |
| shap_guardrail | 0.918919 | 0.801802 | 0 | 0.000000 | 0.000000 | 0.099099 |
| lime_guardrail | 0.990991 | 0.837838 | 0 | 0.000000 | 0.000000 | 0.099099 |
| anchor_guardrail | 0.954955 | 0.819820 | 0 | 0.000000 | 0.000000 | 0.099099 |

- SHAP baseline is included via TreeExplainer.
- LIME baseline is included.
- Anchors baseline is included.
- agreement_proxy is proxy-rule agreement, not clinical expert action accuracy.
- agreement_reference uses independent reference policy (risk + outcome context).