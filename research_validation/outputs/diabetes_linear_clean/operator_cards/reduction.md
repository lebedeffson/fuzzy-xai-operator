# Потери представления

## Input
- `selected_features` = `["f1", "f2", "f3", "f4", "f5"]`
- `feature_importance` = `{"f1": 0.256, "f2": 0.192, "f3": 0.144, "f4": 0.112, "f5": 0.096}`
- `top_k` = `5`

## Formula
delta = 1 - sum(top_k_feature_importance)

## Components
- `selected_features` = `["f1", "f2", "f3", "f4", "f5"]`
- `top_k_importance_sum` = `0.8`
- `calculation` = `"1 - 0.8 = 0.2"`

## Output
- `delta` = `0.2`
- `top_k_importance_sum` = `0.8`

## Thresholds
- `delta_warning` = `0.35`

## Status
passed: Часть объяснения потеряна при top-k редукции.

## Interpretation
Delta показывает, какая доля атрибутивного объяснения не попала в сокращённый набор признаков.

## Next
risk
