# Потери представления

## Input
- `selected_features` = `["f1", "f2", "f3", "f4", "f5"]`
- `feature_importance` = `{"f1": 0.144, "f2": 0.108, "f3": 0.081, "f4": 0.063, "f5": 0.054}`
- `top_k` = `5`

## Formula
delta = 1 - sum(top_k_feature_importance)

## Components
- `selected_features` = `["f1", "f2", "f3", "f4", "f5"]`
- `top_k_importance_sum` = `0.45`
- `calculation` = `"1 - 0.45 = 0.55"`

## Output
- `delta` = `0.55`
- `top_k_importance_sum` = `0.45`

## Thresholds
- `delta_warning` = `0.35`

## Status
warning: Часть объяснения потеряна при top-k редукции.

## Interpretation
Delta показывает, какая доля атрибутивного объяснения не попала в сокращённый набор признаков.

## Next
risk
