# Thesis practice tables

## I_pre / rho quantiles (breast_cancer)

| metric | mean | std | median | p25 | p75 | p05 | p95 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| I_pre | 0.7935691880178595 | 0.028891093213033944 | 0.7913864834269176 | 0.781136104092202 | 0.8049893875841105 | 0.7452511488142374 | 0.8422037308877101 |
| rho | 0.2076536833161616 | 0.15009304671440749 | 0.13846747429895342 | 0.06133269296264607 | 0.34321398144211535 | 0.056597635756142906 | 0.4606185115725768 |

## End-to-end cases

| mode | object | P | representation | Delta | I_pre | rho | chi_R | chi_R_crit | chi_Auto | action |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| accept | sample_0 | 0.015349 | FML-audit | 0.07 | 0.877934 | 0.018382 | 0 | 0 | True | accept |
| audit | sample_18 | 0.144372 | FML-audit | 0.07 | 0.757882 | 0.127603 | 0 | 0 | True | lower_confidence |
| block | sample_2 | 0.322824 | FML-audit | 0.07 | 0.663192 | 0.8 | 1 | 1 | False | block |

## Ablations

| variant | missed_critical_ruptures | false_automatic_accept | false_block_rate | coverage_auto_accept | mean_reduction_loss | agreement_proxy |
| --- | --- | --- | --- | --- | --- | --- |
| without_tau | 1.0 | 0.0 | 0.0 | 0.559441 | 0.0 | 0.699301 |
| without_delta | 0.0 | 0.0 | 0.0 | 0.559441 | 0.0 | 0.944056 |
| without_chi_r_crit | 1.0 | 0.0 | 0.0 | 0.559441 | 0.0 | 0.699301 |
| only_f0 | 0.0 | 0.0 | 0.0 | 0.559441 | 0.25 | 0.93007 |
| without_chi_auto | 0.0 | 0.0 | 0.0 | 0.559441 | 0.0 | 0.944056 |
| probability_threshold_only | 1.0 | 0.132867 | 0.0 | 0.692308 | 0.0 | 0.755245 |

## ExplainPlan parameters

| parameter | value | where_used | selection |
| --- | --- | --- | --- |
| gamma_max | 0.45 | certified alignment / composition | fixed before experiments |
| I_min | 0.5 | interpretability floor | ExplainPlan baseline |
| theta_1...theta_4 | [0.1, 0.25, 0.5, 0.75] | risk observer policy | calibrated policy thresholds |
| w_p,w_u,w_I,w_Delta,w_R | {'predicted_risk': 0.7, 'uncertainty': 0.05, 'reduction_loss': 0.05, 'interpretability_gap': 0.05, 'diagnostic': 0.15} | rho risk aggregation | observer calibration |
| eta_1,eta_2,eta_3 | {'model': 0.5, 'rules': 0.3, 'trace': 0.2} | u_M decomposition | ExplainPlan |
| epsilon_int | 0.001 | interval tolerance / numerical stability | fixed before experiments |
| lambda_Delta | 0.1 | representation reduction penalty | fixed before experiments |