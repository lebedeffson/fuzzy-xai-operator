# Потери представления

## Input
- `selected_features` = `["f1", "f2", "f3", "f4", "f5"]`
- `feature_importance` = `{"f1": 0.2304, "f2": 0.1728, "f3": 0.1296, "f4": 0.1008, "f5": 0.0864}`
- `top_k` = `5`

## Formula
delta = 1 - sum(top_k_feature_importance)

## Components
- `selected_features` = `["f1", "f2", "f3", "f4", "f5"]`
- `top_k_importance_sum` = `0.72`
- `calculation` = `"1 - 0.72 = 0.28"`

## Output
- `delta` = `0.28`
- `top_k_importance_sum` = `0.72`

## Thresholds
- `delta_warning` = `0.35`

## Status
passed: Часть объяснения потеряна при top-k редукции.

## Interpretation
Delta показывает, какая доля атрибутивного объяснения не попала в сокращённый набор признаков.

## Next
risk
