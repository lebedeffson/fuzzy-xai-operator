# Dataset observer report

Dataset: **sklearn_diabetes_binary**
Source: `sklearn.datasets`
Target: `risk_target`
Rows: `442`; columns: `11`
Numeric columns: `11`; categorical columns: `0`
Missing rate: `0.000000`
Suggested uncertainty types: `u_num`
Selected representation: `F0` / `F0`
Selection reason: numeric/linguistic uncertainty is covered by classical fuzzy representation

## Model

- model: `logistic_regression`
- accuracy: `0.774775`
- roc_auc: `0.8340909090909091`

## Observer result

- predicted_risk: `0.552103`
- uncertainty: `0.963245`
- E_M^ext representation: `F0`
- Delta_M: `0.000000`
- I_pre: `0.760918`
- rho: `0.454259`
- action: `request_more_data`
- reason: medium risk with elevated uncertainty
- explanation route: `E_M_ext -> E_R -> E_A`
