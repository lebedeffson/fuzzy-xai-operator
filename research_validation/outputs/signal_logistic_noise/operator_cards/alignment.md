# Согласование T_ij

## Input
- `class_probability` = `0.75`
- `missing_rate` = `0.0`
- `feature_range_violation` = `0.0`
- `conflict_component` = `0.0`
- `interval_width` = `0.0`

## Formula
gamma = max(1 - class_probability, quality_penalty, conflict_component, interval_width)

## Components
- `uncertainty` = `0.25`
- `quality_penalty` = `0.61`
- `conflict_component` = `0.0`
- `interval_width` = `0.0`
- `calculation` = `"max(0.25, 0.61, 0.0, 0.0) = 0.61"`

## Output
- `gamma` = `0.61`

## Thresholds
- `gamma_warning` = `0.35`

## Status
warning: Рассогласование ненулевое, потому что вероятность класса меньше 1.

## Interpretation
Уверенность модели неполная; это ограничивает автоматическое доверие.

## Next
risk
