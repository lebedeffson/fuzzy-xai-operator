# Ablation benchmark: synthetic_ruptures

| mode | agreement_proxy | missed_critical_ruptures | false_auto_accept_rate | false_block_rate | auto_accept_coverage | mean_reduction_loss |
|---|---:|---:|---:|---:|---:|---:|
| full | 0.733333 | 0 | 0.000000 | 0.000000 | 0.164444 | 0.120460 |
| no_trace | 0.622222 | 0 | 0.000000 | 0.000000 | 0.177778 | 0.120460 |
| no_delta | 0.733333 | 0 | 0.000000 | 0.000000 | 0.164444 | 0.000000 |
| no_critical_rupture | 0.422222 | 70 | 0.000000 | 0.000000 | 0.164444 | 0.120460 |
| f0_only | 0.795556 | 0 | 0.000000 | 0.000000 | 0.155556 | 0.250000 |
| no_topos | 0.720000 | 0 | 0.013333 | 0.000000 | 0.177778 | 0.120460 |
| probability_threshold | 0.346667 | 70 | 0.586667 | 0.000000 | 0.764444 | 0.120460 |

- agreement_proxy is a proxy-rule metric, not clinical expert action accuracy.