# Dataset observer report

Dataset: **rikord_demo_fallback**
Source: `fallback:sklearn_diabetes`
Target: `risk_target`
Rows: `442`; columns: `11`
Numeric columns: `11`; categorical columns: `0`
Missing rate: `0.000000`
Suggested uncertainty types: `u_num, u_trace`
Selected representation: `FML-audit` / `FML`
Selection reason: multi-level/audit profile requires preserving several uncertainty types

## Model

- model: `random_forest`
- accuracy: `0.756757`
- roc_auc: `0.8298701298701298`

## Observer result

- predicted_risk: `0.653549`
- uncertainty: `0.859472`
- E_M^ext representation: `FML-audit`
- Delta_M: `0.090000`
- I_pre: `0.753900`
- rho: `0.473653`
- action: `request_more_data`
- reason: medium risk with elevated uncertainty
- explanation route: `E_M_ext -> E_R -> E_A`
