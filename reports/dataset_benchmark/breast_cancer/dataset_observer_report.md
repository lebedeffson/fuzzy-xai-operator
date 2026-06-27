# Dataset observer report

Dataset: **sklearn_breast_cancer**
Source: `sklearn.datasets`
Target: `risk_target`
Rows: `569`; columns: `31`
Numeric columns: `31`; categorical columns: `0`
Missing rate: `0.000000`
Suggested uncertainty types: `u_num, u_trace`
Selected representation: `FML-audit` / `FML`
Selection reason: multi-level/audit profile requires preserving several uncertainty types

## Model

- model: `random_forest`
- accuracy: `0.972028`
- roc_auc: `0.9949685534591195`

## Observer result

- predicted_risk: `0.933060`
- uncertainty: `0.288243`
- E_M^ext representation: `FML-audit`
- Delta_M: `0.090000`
- I_pre: `0.816912`
- rho: `0.402096`
- action: `lower_confidence`
- reason: medium risk: accept only with lower confidence
- explanation route: `E_M_ext -> E_R -> E_A`
