# Data Card: registry_steel_ir

- source: `registry.cit.gov.ru`
- domain: `industrial_cv`
- rows: `8080`
- target_column: `risk_target`
- class_distribution: `n_negative=8015, n_positive=65, positive_rate=0.008045`
- action_labels: `not available`
- applicable_metrics: `model_accuracy, model_roc_auc, rupture_rate, critical_rupture_rate, agreement_proxy`
- recommended_use_in_dissertation: `external-transfer`
- valid_for_quantitative_claims: `False`
- limitations: `['no expert action labels', 'roc_auc near 0.5: ranking signal is weak or class imbalance dominates']`
- notes: `Prototype measurements per object; no I/O timing. Registry mode validates readiness/portability of the pipeline; action quality metric may be N/A. ROC AUC must not be interpreted as quality here; use this mode for industrial contour portability.`