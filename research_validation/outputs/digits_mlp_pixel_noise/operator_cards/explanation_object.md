# Объяснительный объект E_k

## Input
- `class_probability` = `0.66`
- `feature_importance` = `{"pixel1": 0.1888, "pixel2": 0.1416, "pixel3": 0.1062, "pixel4": 0.0826, "pixel5": 0.0708}`

## Formula
E_k = {L_k, μ_k, R_k, α_k, u_k, τ_k}

## Components
- `terms` = `["pixel1", "pixel2", "pixel3", "pixel4", "pixel5"]`
- `uncertainty` = `0.34`

## Output
- `terms` = `["pixel1", "pixel2", "pixel3", "pixel4", "pixel5"]`
- `memberships` = `{"pixel1": 0.1888, "pixel2": 0.1416, "pixel3": 0.1062, "pixel4": 0.0826, "pixel5": 0.0708}`
- `rules` = `[]`
- `activations` = `{"predicted_class": 1}`
- `uncertainty` = `0.34`
- `trace` = `"external_wine_classification:MLPClassifier"`
- `object_summary` = `"табличный объект с вероятностью класса и top-k атрибуциями"`

## Thresholds
n/a

## Status
passed: Объект объяснения содержит вероятность, признаки и атрибуции.

## Interpretation
E_k фиксирует, что модель уверена не полностью и объяснение редуцировано до top-k признаков.

## Next
representation
