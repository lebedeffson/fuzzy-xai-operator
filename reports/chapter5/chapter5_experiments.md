# Chapter 5 experiments

Calibration accuracy: **0.714**

## Calibrated weights

- `predicted_risk`: `0.700`
- `uncertainty`: `0.050`
- `interpretability_gap`: `0.050`
- `reduction_loss`: `0.050`
- `diagnostic`: `0.150`

## scenarios_s0_s6

| scenario | z | uncertainty_type | representation | predicted_risk | uncertainty | i_pre | delta | rupture | expected_action | rho | actual_action | match |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| S0 | -1.2 | none | F0 | 0.12 | 0.1 | 0.92 | 0.02 | False | accept | 0.09399999999999999 | accept | True |
| S1 | 0.1 | model_margin | F0 | 0.4 | 0.52 | 0.86 | 0.04 | False | lower_confidence | 0.31499999999999995 | lower_confidence | True |
| S2 | 0.4 | interval | F_int | 0.45 | 0.3 | 0.74 | 0.42 | False | request_more_data | 0.364 | lower_confidence | False |
| S3 | 0.8 | trace_gap | F_H | 0.58 | 0.34 | 0.42 | 0.18 | False | defer_to_human | 0.46099999999999997 | lower_confidence | False |
| S4 | 0.2 | rupture | F_N_src | 0.33 | 0.4 | 0.55 | 0.3 | True | block | 0.4385 | block | True |
| S5 | 1.9 | high_risk | F0 | 0.88 | 0.3 | 0.65 | 0.1 | False | defer_to_human | 0.6535 | defer_to_human | True |
| S6 | 1.4 | multi_source_conflict | FML-audit | 0.62 | 0.72 | 0.6 | 0.38 | True | block | 0.659 | block | True |

## baseline_comparison

| mode | accuracy | false_blocks | missed_ruptures | n |
| --- | --- | --- | --- | --- |
| risk_threshold | 0.4127142857142857 | 0 | 1963 | 7000 |
| fuzzy_operator | 0.252 | 0 | 1992 | 7000 |
| observer_no_topos | 0.3191428571428571 | 0 | 1000 | 7000 |
| full_fuzzyxai | 0.6128571428571429 | 0 | 0 | 7000 |

## timing_complexity

| config | n | mean_ms | total_s |
| --- | --- | --- | --- |
| predict_only | 1000 | 4.4935000005352776e-05 | 4.4935000005352776e-05 |
| operator_F0 | 1000 | 0.005310757999723137 | 0.005310757999723137 |
| select_F_int | 1000 | 0.005196490999878733 | 0.005196490999878733 |
| select_F_H | 1000 | 0.005157115999281814 | 0.005157115999281814 |
| select_F_N_src | 1000 | 0.0053361860000222805 | 0.0053361860000222805 |
| FML_audit | 1000 | 0.00530588799938414 | 0.00530588799938414 |

## breast_cancer_validation

| dataset | n_test | model_accuracy | model_roc_auc | observer_action_accuracy | simulated_expert_rule |
| --- | --- | --- | --- | --- | --- |
| sklearn_breast_cancer | 143 | 0.951048951048951 | 0.9937106918238994 | 0.8881118881118881 | risk>=0.75 defer; risk>=0.35 or uncertainty>=0.45 lower; ambiguous high-uncertainty cases block |

## context_logic

| object | RiskContext | AutoAccept | restricted_from_E_risk_block |
| --- | --- | --- | --- |
| E_model | ['accept', 'block', 'lower_confidence'] | ['accept', 'lower_confidence'] | block |
| E_risk | ['block', 'request_more_data'] | [] | - |

## sensitivity_w_R

| w_R | accuracy | block_rate |
| --- | --- | --- |
| 0.1 | 0.6322857142857143 | 0.2857142857142857 |
| 0.15 | 0.6064285714285714 | 0.2857142857142857 |
| 0.2 | 0.5652857142857143 | 0.2857142857142857 |
| 0.25 | 0.5484285714285714 | 0.2857142857142857 |
| 0.3 | 0.548 | 0.2857142857142857 |

## sensitivity_theta_high

| theta_high | accuracy | block_rate |
| --- | --- | --- |
| 0.6 | 0.6652857142857143 | 0.2857142857142857 |
| 0.7 | 0.5415714285714286 | 0.2857142857142857 |
| 0.8 | 0.5285714285714286 | 0.2857142857142857 |
| 0.9 | 0.5285714285714286 | 0.2857142857142857 |

## sensitivity_noise

| noise_amp | mean_abs_delta_I | action_change_rate |
| --- | --- | --- |
| 0.0 | 0.0 | 0.0 |
| 0.02 | 0.009867343934207944 | 0.0017142857142857142 |
| 0.05 | 0.024545337848884897 | 0.005142857142857143 |
| 0.1 | 0.048689861371010286 | 0.01042857142857143 |