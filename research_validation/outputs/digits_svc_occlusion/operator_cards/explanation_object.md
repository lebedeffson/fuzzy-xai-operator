# Объяснительный объект E_k

## Input
- `class_probability` = `0.79`
- `feature_importance` = `{"pixel1": 0.224, "pixel2": 0.168, "pixel3": 0.126, "pixel4": 0.098, "pixel5": 0.084}`

## Formula
E_k = {L_k, μ_k, R_k, α_k, u_k, τ_k}

## Components
- `terms` = `["pixel1", "pixel2", "pixel3", "pixel4", "pixel5"]`
- `uncertainty` = `0.21`

## Output
- `terms` = `["pixel1", "pixel2", "pixel3", "pixel4", "pixel5"]`
- `memberships` = `{"pixel1": 0.224, "pixel2": 0.168, "pixel3": 0.126, "pixel4": 0.098, "pixel5": 0.084}`
- `rules` = `[]`
- `activations` = `{"predicted_class": 1}`
- `uncertainty` = `0.21`
- `trace` = `"external_wine_classification:SVC(probability=True)"`
- `object_summary` = `"табличный объект с вероятностью класса и top-k атрибуциями"`

## Thresholds
n/a

## Status
passed: Объект объяснения содержит вероятность, признаки и атрибуции.

## Interpretation
E_k фиксирует, что модель уверена не полностью и объяснение редуцировано до top-k признаков.

## Next
representation
