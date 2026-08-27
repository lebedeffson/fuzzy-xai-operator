# Потеря top-k представления (не Delta)

## Input
- `selected_features` = `["proline", "color_intensity"]`
- `feature_importance` = `{"proline": 0.326532, "color_intensity": 0.243624}`
- `top_k` = `2`

## Formula
presentation_omission_loss = 1 - sum(top_k_feature_importance)

## Components
- `selected_features` = `["proline", "color_intensity"]`
- `top_k_importance_sum` = `0.570156`
- `calculation` = `"1 - 0.570156 = 0.429844"`

## Output
- `presentation_omission_loss` = `0.429844`
- `top_k_importance_sum` = `0.570156`

## Thresholds
- `delta_warning` = `0.35`

## Status
warning: Часть объяснения потеряна при top-k редукции.

## Interpretation
Метрика показывает долю атрибутивного summary вне top-k и не является потерей uncertainty representation.

## Next
legacy_route_score
