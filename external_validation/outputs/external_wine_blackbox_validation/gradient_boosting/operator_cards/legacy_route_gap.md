# Legacy route gap (не Gamma)

## Input
- `class_probability` = `0.679119`
- `missing_rate` = `0.0`
- `feature_range_violation` = `0.0`
- `conflict_component` = `0.0`
- `interval_width` = `0.0`

## Formula
legacy_route_gap = max(1 - class_probability, quality_penalty, conflict_component, interval_width)

## Components
- `uncertainty` = `0.320881`
- `quality_penalty` = `0.0`
- `conflict_component` = `0.0`
- `interval_width` = `0.0`
- `calculation` = `"max(0.320881, 0.0, 0.0, 0.0) = 0.320881"`

## Output
- `legacy_route_gap` = `0.320881`

## Thresholds
- `gamma_warning` = `0.35`

## Status
passed: Рассогласование ненулевое, потому что вероятность класса меньше 1.

## Interpretation
Уверенность модели неполная; это ограничивает автоматическое доверие.

## Next
legacy_route_score
