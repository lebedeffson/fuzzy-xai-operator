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
external_validation/outputs/external_wine_logistic_regression_route.json
external_validation/outputs/external_wine_logistic_regression_proof_trace.json
external_validation/outputs/external_wine_logistic_regression_operator_dashboard.png
external_validation/outputs/external_wine_logistic_regression_summary.json
external_validation/outputs/external_wine_gradient_boosting_route.json
external_validation/outputs/external_wine_gradient_boosting_proof_trace.json
external_validation/outputs/external_wine_gradient_boosting_operator_dashboard.png
external_validation/outputs/external_wine_gradient_boosting_summary.json
external_validation/outputs/external_wine_blackbox_validation.zip
```
