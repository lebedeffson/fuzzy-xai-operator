# Data Card: diabetes_binary

- source: `sklearn.datasets`
- domain: `medical`
- rows: `442`
- target_column: `risk_target`
- class_distribution: `n_negative=221, n_positive=221, positive_rate=0.500000`
- action_labels: `available`
- applicable_metrics: `model_accuracy, model_roc_auc, rupture_rate, critical_rupture_rate, agreement_proxy`
- recommended_use_in_dissertation: `stress-test`
- valid_for_quantitative_claims: `False`
- limitations: `['no expert action labels']`
- notes: `Prototype measurements per object; no I/O timing. Stress-test for borderline uncertainty; threshold calibration may be required.`