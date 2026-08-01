# H10-C4 Cost Models

All cost models describe machine-executable repair operations. They are not
proxies for engineer labor or organizational expense.

| ID | Formula | Interpretation | Limitation |
|---|---|---|---|
| M0 Uniform | `1` | Executed action count | Ignores action differences |
| M1 Runtime | development-normalized measured runtime | Machine execution cost | Hardware-dependent and not human time |
| M2 Dependency | `1 + log(1 + fanout)` | Structural impact | Does not measure organizational impact |
| M3 Hybrid | `1 + alpha*t + beta*log(1+d) + gamma*r` | Registered sensitivity analysis | Depends on preregistered weights |

Runtime, dependency fan-out, and rollback/side-effect risk normalization are
fitted only on the 24 development scenarios. Held-out outcomes are not used to
set scales or weights.

The primary model is the hybrid model with `alpha = beta = gamma = 0.5`.
Primary executable cost is divided by the `B_ALL` cost within each scenario.

The hybrid grid is fixed before execution:

```text
alpha in {0.0, 0.25, 0.5, 1.0}
beta  in {0.0, 0.25, 0.5, 1.0}
gamma in {0.0, 0.5, 1.0}
```

Selection stability passes when the nominal cut or its equivalent optimal-cut
class is selected in at least 80% of the 48 configurations.
