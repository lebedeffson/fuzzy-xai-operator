# Dataset observer report

Dataset: **synthetic_ruptures**
Source: `fuzzyxai.synthetic`
Target: `risk_target`
Rows: `900`; columns: `8`
Numeric columns: `8`; categorical columns: `0`
Missing rate: `0.000000`
Suggested uncertainty types: `u_conf, u_expert, u_num`
Selected representation: `FML-user` / `FML`
Selection reason: multi-level/audit profile requires preserving several uncertainty types

## Model

- model: `random_forest`
- accuracy: `0.951111`
- roc_auc: `0.9955908289241623`

## Observer result

- predicted_risk: `0.040296`
- uncertainty: `0.194731`
- E_M^ext representation: `FML-user`
- Delta_M: `0.116716`
- I_pre: `0.774495`
- rho: `0.123380`
- action: `accept`
- reason: risk score below theta_mid
- explanation route: `E_M_ext -> E_R -> E_A`
