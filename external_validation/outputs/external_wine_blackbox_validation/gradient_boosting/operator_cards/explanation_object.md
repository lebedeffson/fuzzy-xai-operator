# Объяснительный объект E_k

## Input
- `class_probability` = `0.679119`
- `feature_importance` = `{"proline": 0.326532, "color_intensity": 0.243624}`

## Formula
E_k = {L_k, μ_k, R_k, α_k, u_k, τ_k}

## Components
- `terms` = `["proline", "color_intensity"]`
- `uncertainty` = `0.320881`

## Output
- `terms` = `["proline", "color_intensity"]`
- `memberships` = `{"proline": 0.326532, "color_intensity": 0.243624}`
- `rules` = `[]`
- `activations` = `{"predicted_class": 0}`
- `uncertainty` = `0.320881`
- `trace` = `"external_wine_classification:GradientBoostingClassifier"`
- `object_summary` = `"табличный объект с вероятностью класса и top-k атрибуциями"`

## Thresholds
n/a

## Status
passed: Объект объяснения содержит вероятность, признаки и атрибуции.

## Interpretation
E_k фиксирует, что модель уверена не полностью и объяснение редуцировано до top-k признаков.

## Next
representation
