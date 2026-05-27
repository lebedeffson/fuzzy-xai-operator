# Data Card: registry_programs

- source: `registry.cit.gov.ru`
- domain: `education_science_industry`
- rows: `10007`
- target_column: `risk_target`
- class_distribution: `n_negative=5004, n_positive=5003, positive_rate=0.499950`
- action_labels: `not available`
- applicable_metrics: `model_accuracy, model_roc_auc, rupture_rate, critical_rupture_rate, observer_action_proxy_accuracy`
- recommended_use_in_dissertation: `external-transfer`
- valid_for_quantitative_claims: `False`
- limitations: `['no expert action labels', 'roc_auc near 0.5: ranking signal is weak or class imbalance dominates']`
- notes: `Prototype measurements per object; no I/O timing. Registry mode validates readiness/portability of the pipeline; action quality metric may be N/A. No expert action labels: observer_action_accuracy is not applicable.`