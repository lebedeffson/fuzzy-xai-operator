# Data Card: breast_cancer

- source: `sklearn.datasets`
- domain: `medical`
- rows: `569`
- target_column: `risk_target`
- class_distribution: `n_negative=357, n_positive=212, positive_rate=0.372583`
- action_labels: `available`
- applicable_metrics: `model_accuracy, model_roc_auc, rupture_rate, critical_rupture_rate, agreement_proxy`
- recommended_use_in_dissertation: `quantitative`
- valid_for_quantitative_claims: `True`
- limitations: `['no expert action labels']`
- notes: `Prototype measurements per object; no I/O timing.`