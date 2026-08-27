# External FuzzyXAI Black-Box Validation

- task: `sklearn_wine_classification`
- scenario_id: `external_wine_classification`
- source_commit: `97074b57a5522cb1cd7150f288a8b27bd915b357`
- verifier: `passed`
- package: `external_wine_blackbox_validation.zip`

The package was generated from an installed `fuzzyxai` framework import and does not use `applications/scenarios`.
Both checks use moderate-confidence wine-classification objects and top-k feature importances, so operator values are non-zero.

| Model | p | legacy route gap | presentation omission | legacy route score | action | diagnostic |
|---|---:|---:|---:|---:|---|---|
| LogisticRegression | 0.689724 | 0.310276 | 0.373128 | 0.373128 | lower_confidence | D_external_tabular_uncertainty |
| GradientBoostingClassifier | 0.679119 | 0.320881 | 0.429844 | 0.429844 | lower_confidence | D_external_tabular_uncertainty |

Formulas:

- `legacy_route_gap = max(1 - class_probability, quality_penalty, conflict, interval)`; no T_ij, therefore not Gamma
- `presentation_omission_loss = 1 - sum(top_k_feature_importance)`; not dissertation Delta
- `legacy_route_score = max(legacy route metrics)`; not dissertation rho
