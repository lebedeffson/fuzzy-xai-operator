# Потери представления

## Input
- `selected_features` = `["f1", "f2", "f3", "f4", "f5"]`
- `feature_importance` = `{"f1": 0.2368, "f2": 0.1776, "f3": 0.1332, "f4": 0.1036, "f5": 0.0888}`
- `top_k` = `5`

## Formula
delta = 1 - sum(top_k_feature_importance)

## Components
- `selected_features` = `["f1", "f2", "f3", "f4", "f5"]`
- `top_k_importance_sum` = `0.74`
- `calculation` = `"1 - 0.74 = 0.26"`

## Output
- `delta` = `0.26`
- `top_k_importance_sum` = `0.74`

## Thresholds
- `delta_warning` = `0.35`

## Status
passed: Часть объяснения потеряна при top-k редукции.

## Interpretation
Delta показывает, какая доля атрибутивного объяснения не попала в сокращённый набор признаков.

## Next
risk
