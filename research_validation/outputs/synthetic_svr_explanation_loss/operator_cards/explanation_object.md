# Объяснительный объект E_k

## Input
- `class_probability` = `0.71`
- `feature_importance` = `{"f1": 0.144, "f2": 0.108, "f3": 0.081, "f4": 0.063, "f5": 0.054}`

## Formula
E_k = {L_k, μ_k, R_k, α_k, u_k, τ_k}

## Components
- `terms` = `["f1", "f2", "f3", "f4", "f5"]`
- `uncertainty` = `0.29`

## Output
- `terms` = `["f1", "f2", "f3", "f4", "f5"]`
- `memberships` = `{"f1": 0.144, "f2": 0.108, "f3": 0.081, "f4": 0.063, "f5": 0.054}`
- `rules` = `[]`
- `activations` = `{"predicted_class": 1}`
- `uncertainty` = `0.29`
- `trace` = `"external_wine_classification:SVR"`
- `object_summary` = `"табличный объект с вероятностью класса и top-k атрибуциями"`

## Thresholds
n/a

## Status
passed: Объект объяснения содержит вероятность, признаки и атрибуции.

## Interpretation
E_k фиксирует, что модель уверена не полностью и объяснение редуцировано до top-k признаков.

## Next
representation
