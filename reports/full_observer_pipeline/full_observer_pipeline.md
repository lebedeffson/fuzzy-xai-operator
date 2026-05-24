# Full observer pipeline

Status: **PASS**
Layer: `active risk-oriented observer over a predictive interface`

## Core claim

observer does not modify model parameters; it determines the admissible mode of prediction use

## Route

data -> model prediction -> E_M_ext -> A_M^F -> E_R -> E_D -> E_G -> RiskPolicy -> safe_action

## Case result

- raw prediction: `1`
- risk score r_M(x): `0.924851`
- uncertainty u_M(x): `0.314484`
- selected A_M^F: `FML-audit` / `FML`
- Delta_M: `0.196250`
- I(E_G): `0.707878`
- rho(x): `0.443938`
- safe action: `lower_confidence`
- reason: medium risk: accept only with lower confidence

## Without / with observer

- Without observer: model probability only.
- With observer: E_M^ext, uncertainty, Delta, gamma, I(E_G), diagnostics and safe action.

## Composition edges

- `Model -> RiskModule`: gamma=`0.443873`, severity=`orange`
- `RiskModule -> Decision`: gamma=`0.350401`, severity=`orange`
