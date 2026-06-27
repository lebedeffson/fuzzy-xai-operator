# Data Card: registry_mosmed_doctor_analysis

- source: `registry.cit.gov.ru`
- domain: `medical_audit`
- rows: `10`
- target_column: `risk_target`
- class_distribution: `n_negative=9, n_positive=1, positive_rate=0.100000`
- action_labels: `not available`
- applicable_metrics: `model_accuracy, model_roc_auc, rupture_rate, critical_rupture_rate, agreement_proxy`
- recommended_use_in_dissertation: `external-transfer`
- valid_for_quantitative_claims: `False`
- limitations: `['no expert action labels', 'roc_auc undefined: single class in test split', 'small sample size']`
- notes: `Prototype measurements per object; no I/O timing. Registry mode validates readiness/portability of the pipeline; action quality metric may be N/A. Small audit slice (n=10): not for statistical validation.`