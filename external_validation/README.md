# External FuzzyXAI Validation

This folder contains a black-box validation task for the installable FuzzyXAI
framework. It does not import `applications/scenarios` and does not use the
internal HYBRID-XIRIS/ECG/GD/BEACON/GIS demo scenarios.

Task:

```text
sklearn wine classification
  -> LogisticRegression
  -> GradientBoostingClassifier
  -> generic tabular adapter
  -> FuzzyXAI public API
  -> route/proof/dashboard
```

Both external checks select a moderate-confidence object and pass only top-k
feature importances, so `gamma`, `delta` and `rho` must be non-zero.

Run:

```bash
python external_validation/run_external_wine_test.py
```

Outputs:

```text
external_validation/outputs/external_wine_summary.json
external_validation/outputs/external_wine_blackbox_validation.zip
external_validation/outputs/external_wine_blackbox_validation/
  manifest.json
  external_validation_report.md
  import_provenance.json
  external_wine_summary.json
  logistic_regression/
    route.json
    operator_trace.json
    operator_table.csv
    proof_trace.json
    verifier_report.json
    dashboard_data.json
    summary.json
    operator_dashboard.png
    operator_dashboard_v2.png
    operator_dashboard_v2.html
    operator_cards/
  gradient_boosting/
    route.json
    operator_trace.json
    operator_table.csv
    proof_trace.json
    verifier_report.json
    dashboard_data.json
    summary.json
    operator_dashboard.png
    operator_dashboard_v2.png
    operator_dashboard_v2.html
    operator_cards/
```
