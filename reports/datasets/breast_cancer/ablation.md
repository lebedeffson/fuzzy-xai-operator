# Ablation benchmark: breast_cancer

| mode | agreement_proxy | missed_critical_ruptures | false_auto_accept_rate | false_block_rate | auto_accept_coverage | mean_reduction_loss |
|---|---:|---:|---:|---:|---:|---:|
| full | 0.650350 | 0 | 0.000000 | 0.000000 | 0.517483 | 0.078790 |
| no_trace | 0.650350 | 0 | 0.000000 | 0.000000 | 0.517483 | 0.078790 |
| no_delta | 0.657343 | 0 | 0.000000 | 0.000000 | 0.517483 | 0.000000 |
| no_critical_rupture | 0.650350 | 0 | 0.000000 | 0.000000 | 0.517483 | 0.078790 |
| f0_only | 0.629371 | 0 | 0.000000 | 0.000000 | 0.517483 | 0.250000 |
| no_topos | 0.601399 | 0 | 0.048951 | 0.000000 | 0.566434 | 0.078790 |
| probability_threshold | 0.804196 | 0 | 0.195804 | 0.000000 | 0.713287 | 0.078790 |

- agreement_proxy is a proxy-rule metric, not clinical expert action accuracy.