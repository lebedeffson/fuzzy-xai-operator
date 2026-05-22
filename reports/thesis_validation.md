# Thesis numerical validation report

Status: **PASS**
Total checks: 20
Failed checks: 0

## Checked dissertation routes

- Chapter 2: `z -> mu -> H -> d_E -> composition -> L -> I -> D_ij`.
- Chapter 3: `metadata -> P_sit -> F* -> A_k^F -> Delta -> E_k_ext -> d_E_ext -> D_choice`.

## Checks

| Check | Expected | Actual | Tolerance | Status |
|---|---:|---:|---:|---|
| `chapter2.risk` | `0.72` | `0.72` | `1e-06` | PASS |
| `chapter2.mu_low` | `0.0` | `0.0` | `1e-06` | PASS |
| `chapter2.mu_medium` | `0.12` | `0.12` | `1e-09` | PASS |
| `chapter2.mu_high` | `0.88` | `0.88` | `1e-09` | PASS |
| `chapter2.entropy_H` | `0.342207` | `0.342207` | `5e-06` | PASS |
| `chapter2.gamma_d_E` | `0.36108` | `0.36108` | `5e-06` | PASS |
| `chapter2.composed_uncertainty` | `0.449813` | `0.449813` | `5e-06` | PASS |
| `chapter2.loss_L` | `0.284404` | `0.284404` | `5e-06` | PASS |
| `chapter2.index_I` | `0.752463` | `0.752463` | `5e-06` | PASS |
| `chapter2.diagnostic_code` | `D_ij` | `D_ij` | `None` | PASS |
| `chapter3.profile` | `['u_conf', 'u_exp', 'u_int', 'u_ling', 'u_multi', 'u_num', 'u_trace']` | `['u_conf', 'u_exp', 'u_int', 'u_ling', 'u_multi', 'u_num', 'u_trace']` | `None` | PASS |
| `chapter3.selected_class` | `FML-audit` | `FML-audit` | `None` | PASS |
| `chapter3.delta_interval` | `0.04` | `0.04` | `1e-09` | PASS |
| `chapter3.delta_hesitant` | `0.085` | `0.085` | `1e-09` | PASS |
| `chapter3.delta_neutrosophic` | `0.26` | `0.26` | `1e-09` | PASS |
| `chapter3.delta_multilevel` | `0.20125` | `0.20125` | `1e-09` | PASS |
| `chapter3.extended_d_E` | `0.163045` | `0.163045` | `5e-06` | PASS |
| `chapter3.composition_result` | `ExplanationObject` | `ExplanationObject` | `None` | PASS |
| `chapter3.diagnostic_choice_code` | `D_choice` | `D_choice` | `None` | PASS |
| `chapter3.diagnostic_missing` | `['u_ethical']` | `['u_ethical']` | `None` | PASS |
