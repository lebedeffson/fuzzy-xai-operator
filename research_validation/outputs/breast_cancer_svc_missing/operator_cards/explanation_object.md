# Объяснительный объект E_k

## Input
- `class_probability` = `0.72`
- `feature_importance` = `{"f1": 0.224, "f2": 0.168, "f3": 0.126, "f4": 0.098, "f5": 0.084}`

## Formula
E_k = {L_k, μ_k, R_k, α_k, u_k, τ_k}

## Components
- `terms` = `["f1", "f2", "f3", "f4", "f5"]`
- `uncertainty` = `0.28`

## Output
- `terms` = `["f1", "f2", "f3", "f4", "f5"]`
- `memberships` = `{"f1": 0.224, "f2": 0.168, "f3": 0.126, "f4": 0.098, "f5": 0.084}`
- `rules` = `[]`
- `activations` = `{"predicted_class": 1}`
- `uncertainty` = `0.28`
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
