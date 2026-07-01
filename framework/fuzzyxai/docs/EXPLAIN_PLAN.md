# ExplainPlan

`ExplainPlan` stores thresholds and policies used by the runtime.

Default fields include:

```json
{
  "gamma_warning": 0.25,
  "gamma_critical": 0.60,
  "delta_warning": 0.25,
  "delta_critical": 0.60,
  "rho_accept": 0.35,
  "rho_warning": 0.60,
  "rho_audit": 0.75,
  "rho_critical": 0.85,
  "top_k": 5,
  "representation_policy": "auto",
  "action_policy": "risk_zone"
}
```

Use:

```python
from fuzzyxai import ExplainPlan, FuzzyXAI

plan = ExplainPlan.default()
fxai = FuzzyXAI(plan=plan)
```
