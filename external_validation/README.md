# External FuzzyXAI Validation

This folder contains a black-box validation task for the installable FuzzyXAI
framework. It does not import `applications/scenarios` and does not use the
internal HYBRID-XIRIS/ECG/GD/BEACON/GIS demo scenarios.

Task:

```text
sklearn wine classification -> RandomForestClassifier -> generic tabular adapter
-> FuzzyXAI public API -> route/proof/dashboard
```

Run:

```bash
python external_validation/run_external_wine_test.py
```

Outputs:

```text
external_validation/outputs/external_wine_route.json
external_validation/outputs/external_wine_proof_trace.json
external_validation/outputs/external_wine_operator_dashboard.png
external_validation/outputs/external_wine_summary.json
```
