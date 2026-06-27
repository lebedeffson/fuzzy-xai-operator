# Data Card: synthetic_ruptures

- source: `fuzzyxai.synthetic`
- domain: `controlled`
- rows: `900`
- target_column: `risk_target`
- class_distribution: `n_negative=502, n_positive=398, positive_rate=0.442222`
- action_labels: `available`
- applicable_metrics: `model_accuracy, model_roc_auc, rupture_rate, critical_rupture_rate, agreement_proxy`
- recommended_use_in_dissertation: `control-diagnostics`
- valid_for_quantitative_claims: `True`
- limitations: `[]`
- notes: `Prototype measurements per object; no I/O timing. Rupture proxies are derived from expert/source disagreement fields.`