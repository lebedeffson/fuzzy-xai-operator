# Data Card: wine_risk

- source: `sklearn.datasets`
- domain: `general`
- rows: `178`
- target_column: `risk_target`
- class_distribution: `n_negative=130, n_positive=48, positive_rate=0.269663`
- action_labels: `available`
- applicable_metrics: `model_accuracy, model_roc_auc, rupture_rate, critical_rupture_rate, agreement_proxy`
- recommended_use_in_dissertation: `quantitative`
- valid_for_quantitative_claims: `True`
- limitations: `['no expert action labels']`
- notes: `Prototype measurements per object; no I/O timing.`