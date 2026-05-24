# Full observer pipeline

Status: **PASS**
Layer: `active risk-oriented observer over a predictive interface`

## Core claim

observer does not modify model parameters; it determines the admissible mode of prediction use

## Route

data -> model prediction -> E_M_ext -> A_M^F -> E_pre -> I_pre -> rho -> safe_action -> E_A -> E_G -> I_final

## Case result

- raw prediction: `1`
- risk score r_M(x): `0.924851`
- uncertainty u_M(x): `0.314484`
- selected A_M^F: `FML-audit` / `FML`
- Delta_M: `0.196250`
- I_pre: `0.801337`
- rho(x): `0.425247`
- I_final: `0.707878`
- safe action: `lower_confidence`
- reason: medium risk: accept only with lower confidence

## Without / with observer

- Without observer: model probability only.
- With observer: E_M^ext, uncertainty, Delta, I_pre, rho(x), I_final, diagnostics and safe action.

## Composition edges

- `Model -> RiskModule`: gamma=`0.443873`, severity=`orange`
- `RiskModule -> Action`: gamma=`0.350401`, severity=`orange`
