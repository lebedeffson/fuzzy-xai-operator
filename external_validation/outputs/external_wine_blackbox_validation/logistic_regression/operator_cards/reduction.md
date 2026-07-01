# Потери представления

## Input
- `selected_features` = `["alcohol", "proline", "color_intensity", "ash", "hue"]`
- `feature_importance` = `{"alcohol": 0.166384, "proline": 0.145116, "color_intensity": 0.133733, "ash": 0.0985, "hue": 0.083139}`
- `top_k` = `5`

## Formula
delta = 1 - sum(top_k_feature_importance)

## Components
- `selected_features` = `["alcohol", "proline", "color_intensity", "ash", "hue"]`
- `top_k_importance_sum` = `0.626872`
- `calculation` = `"1 - 0.626872 = 0.373128"`

## Output
- `delta` = `0.373128`
- `top_k_importance_sum` = `0.626872`

## Thresholds
- `delta_warning` = `0.35`

## Status
warning: Часть объяснения потеряна при top-k редукции.

## Interpretation
Delta показывает, какая доля атрибутивного объяснения не попала в сокращённый набор признаков.

## Next
risk
