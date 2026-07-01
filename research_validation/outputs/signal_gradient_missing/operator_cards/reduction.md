# Потери представления

## Input
- `selected_features` = `["f1", "f2", "f3", "f4", "f5"]`
- `feature_importance` = `{"f1": 0.2176, "f2": 0.1632, "f3": 0.1224, "f4": 0.0952, "f5": 0.0816}`
- `top_k` = `5`

## Formula
delta = 1 - sum(top_k_feature_importance)

## Components
- `selected_features` = `["f1", "f2", "f3", "f4", "f5"]`
- `top_k_importance_sum` = `0.68`
- `calculation` = `"1 - 0.68 = 0.32"`

## Output
- `delta` = `0.32`
- `top_k_importance_sum` = `0.68`

## Thresholds
- `delta_warning` = `0.35`

## Status
passed: Часть объяснения потеряна при top-k редукции.

## Interpretation
Delta показывает, какая доля атрибутивного объяснения не попала в сокращённый набор признаков.

## Next
risk
