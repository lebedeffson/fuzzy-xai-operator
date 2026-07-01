# Потери представления

## Input
- `selected_features` = `["pixel1", "pixel2", "pixel3", "pixel4", "pixel5"]`
- `feature_importance` = `{"pixel1": 0.2592, "pixel2": 0.1944, "pixel3": 0.1458, "pixel4": 0.1134, "pixel5": 0.0972}`
- `top_k` = `5`

## Formula
delta = 1 - sum(top_k_feature_importance)

## Components
- `selected_features` = `["pixel1", "pixel2", "pixel3", "pixel4", "pixel5"]`
- `top_k_importance_sum` = `0.81`
- `calculation` = `"1 - 0.81 = 0.19"`

## Output
- `delta` = `0.19`
- `top_k_importance_sum` = `0.81`

## Thresholds
- `delta_warning` = `0.35`

## Status
passed: Часть объяснения потеряна при top-k редукции.

## Interpretation
Delta показывает, какая доля атрибутивного объяснения не попала в сокращённый набор признаков.

## Next
risk
