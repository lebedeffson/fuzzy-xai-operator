# Потери представления

## Input
- `selected_features` = `["pixel1", "pixel2", "pixel3", "pixel4", "pixel5"]`
- `feature_importance` = `{"pixel1": 0.16, "pixel2": 0.12, "pixel3": 0.09, "pixel4": 0.07, "pixel5": 0.06}`
- `top_k` = `5`

## Formula
delta = 1 - sum(top_k_feature_importance)

## Components
- `selected_features` = `["pixel1", "pixel2", "pixel3", "pixel4", "pixel5"]`
- `top_k_importance_sum` = `0.5`
- `calculation` = `"1 - 0.5 = 0.5"`

## Output
- `delta` = `0.5`
- `top_k_importance_sum` = `0.5`

## Thresholds
- `delta_warning` = `0.35`

## Status
warning: Часть объяснения потеряна при top-k редукции.

## Interpretation
Delta показывает, какая доля атрибутивного объяснения не попала в сокращённый набор признаков.

## Next
risk
