# FuzzyXAI API

Core public API:

```python
from fuzzyxai import FuzzyXAI, ExplainPlan
from fuzzyxai.adapters import get_adapter

payload = {...}
plan = ExplainPlan.default()
fxai = FuzzyXAI(plan=plan)
adapter = get_adapter("tabular_classification")

route = fxai.run_payload(payload, adapter)
fxai.export_package(route, "out")
```

Function API remains available:

```python
from fuzzyxai import build_route, build_proof_trace, verify_proof_trace, render_dashboard
```

`FuzzyXAI.export_package(...)` writes `route.json`, `proof_trace.json`,
`operator_trace.json`, `operator_table.csv`, `verifier_report.json`,
`dashboard_data.json`, `operator_dashboard.png`, `operator_dashboard_v2.png`,
`operator_dashboard_v2.html` and `operator_cards/`.
