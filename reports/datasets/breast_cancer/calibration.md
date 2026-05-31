# Observer calibration: breast_cancer

| Mode | agreement_proxy | missed_critical_ruptures | false_auto_accept_rate |
|---|---:|---:|---:|
| before | 0.706294 | 0 | 0.000000 |
| after | 0.958042 | 0 | 0.000000 |

| Mode | agreement_reference | missed_critical_ruptures | false_auto_accept_rate |
|---|---:|---:|---:|
| before | 0.734266 | 0 | 0.000000 |
| after | 0.930070 | 0 | 0.000000 |

- thresholds: `(0.12, 0.28, 0.52, 0.8)`
- gamma_max: `0.4`
- i_min: `0.6`
- delta_max: `0.12`
- uncertainty_max: `0.4`
- weights: `{'predicted_risk': 0.5, 'uncertainty': 0.2, 'interpretability_gap': 0.1, 'reduction_loss': 0.1, 'diagnostic': 0.1}`

- agreement_proxy is a proxy-rule metric, not clinical expert action accuracy.