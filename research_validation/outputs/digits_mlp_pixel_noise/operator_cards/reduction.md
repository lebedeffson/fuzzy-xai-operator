# Потери представления

## Input
- `selected_features` = `["pixel1", "pixel2", "pixel3", "pixel4", "pixel5"]`
- `feature_importance` = `{"pixel1": 0.1888, "pixel2": 0.1416, "pixel3": 0.1062, "pixel4": 0.0826, "pixel5": 0.0708}`
- `top_k` = `5`

## Formula
delta = 1 - sum(top_k_feature_importance)

## Components
- `selected_features` = `["pixel1", "pixel2", "pixel3", "pixel4", "pixel5"]`
- `top_k_importance_sum` = `0.59`
- `calculation` = `"1 - 0.59 = 0.41"`

## Output
- `delta` = `0.41`
- `top_k_importance_sum` = `0.59`

## Thresholds
- `delta_warning` = `0.35`

## Status
warning: Часть объяснения потеряна при top-k редукции.

## Interpretation
Delta показывает, какая доля атрибутивного объяснения не попала в сокращённый набор признаков.

## Next
risk
