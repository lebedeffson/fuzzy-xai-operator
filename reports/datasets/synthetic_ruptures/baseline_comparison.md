# Baseline comparison: synthetic_ruptures

- baseline_access: `equal_guardrail`

| baseline | information_access | agreement_proxy | agreement_reference | missed_critical_ruptures | false_auto_accept_rate | false_block_rate | auto_accept_coverage |
|---|---|---:|---:|---:|---:|---:|---:|
| full_observer_runtime | full_structure | 0.760000 | 0.835556 | 0 | 0.000000 | 0.000000 | 0.151111 |
| full_observer_uncalibrated | full_structure | 0.773333 | 0.822222 | 0 | 0.000000 | 0.000000 | 0.151111 |
| full_observer_calibrated | full_structure | 0.773333 | 0.822222 | 0 | 0.000000 | 0.000000 | 0.151111 |
| probability_threshold | equal_guardrail | 1.000000 | 0.871111 | 0 | 0.000000 | 0.000000 | 0.168889 |
| shap_guardrail | equal_guardrail | 0.986667 | 0.857778 | 0 | 0.000000 | 0.000000 | 0.168889 |
| lime_guardrail | equal_guardrail | 0.995556 | 0.866667 | 0 | 0.000000 | 0.000000 | 0.168889 |
| anchor_guardrail | equal_guardrail | 0.982222 | 0.853333 | 0 | 0.000000 | 0.000000 | 0.168889 |

- SHAP baseline is included via TreeExplainer.
- LIME baseline is included.
- Anchors baseline is included.
- agreement_proxy is proxy-rule agreement, not clinical expert action accuracy.
- agreement_reference uses independent reference policy (risk + outcome context).