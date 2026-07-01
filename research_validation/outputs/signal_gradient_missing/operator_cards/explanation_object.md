# Объяснительный объект E_k

## Input
- `class_probability` = `0.8`
- `feature_importance` = `{"f1": 0.2176, "f2": 0.1632, "f3": 0.1224, "f4": 0.0952, "f5": 0.0816}`

## Formula
E_k = {L_k, μ_k, R_k, α_k, u_k, τ_k}

## Components
- `terms` = `["f1", "f2", "f3", "f4", "f5"]`
- `uncertainty` = `0.2`

## Output
- `terms` = `["f1", "f2", "f3", "f4", "f5"]`
- `memberships` = `{"f1": 0.2176, "f2": 0.1632, "f3": 0.1224, "f4": 0.0952, "f5": 0.0816}`
- `rules` = `[]`
- `activations` = `{"predicted_class": 1}`
- `uncertainty` = `0.2`
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
