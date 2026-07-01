# Объяснительный объект E_k

## Input
- `class_probability` = `0.7`
- `feature_importance` = `{"pixel1": 0.16, "pixel2": 0.12, "pixel3": 0.09, "pixel4": 0.07, "pixel5": 0.06}`

## Formula
E_k = {L_k, μ_k, R_k, α_k, u_k, τ_k}

## Components
- `terms` = `["pixel1", "pixel2", "pixel3", "pixel4", "pixel5"]`
- `uncertainty` = `0.3`

## Output
- `terms` = `["pixel1", "pixel2", "pixel3", "pixel4", "pixel5"]`
- `memberships` = `{"pixel1": 0.16, "pixel2": 0.12, "pixel3": 0.09, "pixel4": 0.07, "pixel5": 0.06}`
- `rules` = `[]`
- `activations` = `{"predicted_class": 1}`
- `uncertainty` = `0.3`
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
