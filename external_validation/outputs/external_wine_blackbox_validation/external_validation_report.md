# External FuzzyXAI Black-Box Validation

- task: `sklearn_wine_classification`
- scenario_id: `external_wine_classification`
- source_commit: `be2a30096b275d3365a92cb69b165e57e99e90de`
- verifier: `passed`
- package: `external_wine_blackbox_validation.zip`

The package was generated from an installed `fuzzyxai` framework import and does not use `applications/scenarios`.
Both checks use moderate-confidence wine-classification objects and top-k feature importances, so operator values are non-zero.

| Model | p | gamma | delta | rho | action | diagnostic |
|---|---:|---:|---:|---:|---|---|
| LogisticRegression | 0.689724 | 0.310276 | 0.373128 | 0.373128 | lower_confidence | D_external_tabular_uncertainty |
| GradientBoostingClassifier | 0.679119 | 0.320881 | 0.429844 | 0.429844 | lower_confidence | D_external_tabular_uncertainty |

Formulas:

- `gamma = max(1 - class_probability, quality_penalty)`
- `delta = 1 - sum(top_k_feature_importance)`
- `rho = max(gamma, delta)`
