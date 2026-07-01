# Объяснительный объект E_k

## Input
- `class_probability` = `0.689724`
- `feature_importance` = `{"alcohol": 0.166384, "proline": 0.145116, "color_intensity": 0.133733, "ash": 0.0985, "hue": 0.083139}`

## Formula
E_k = {L_k, μ_k, R_k, α_k, u_k, τ_k}

## Components
- `terms` = `["alcohol", "proline", "color_intensity", "ash", "hue"]`
- `uncertainty` = `0.310276`

## Output
- `terms` = `["alcohol", "proline", "color_intensity", "ash", "hue"]`
- `memberships` = `{"alcohol": 0.166384, "proline": 0.145116, "color_intensity": 0.133733, "ash": 0.0985, "hue": 0.083139}`
- `rules` = `[]`
- `activations` = `{"predicted_class": 1}`
- `uncertainty` = `0.310276`
- `trace` = `"external_wine_classification:LogisticRegression"`
- `object_summary` = `"табличный объект с вероятностью класса и top-k атрибуциями"`

## Thresholds
n/a

## Status
passed: Объект объяснения содержит вероятность, признаки и атрибуции.

## Interpretation
E_k фиксирует, что модель уверена не полностью и объяснение редуцировано до top-k признаков.

## Next
representation
