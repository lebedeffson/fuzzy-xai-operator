# Согласование T_ij

## Input
- `class_probability` = `0.74`
- `missing_rate` = `0.52`
- `feature_range_violation` = `0.0`
- `conflict_component` = `0.0`
- `interval_width` = `0.36`

## Formula
gamma = max(1 - class_probability, quality_penalty, conflict_component, interval_width)

## Components
- `uncertainty` = `0.26`
- `quality_penalty` = `0.52`
- `conflict_component` = `0.0`
- `interval_width` = `0.36`
- `calculation` = `"max(0.26, 0.52, 0.0, 0.36) = 0.52"`

## Output
- `gamma` = `0.52`

## Thresholds
- `gamma_warning` = `0.35`

## Status
warning: Рассогласование ненулевое, потому что вероятность класса меньше 1.

## Interpretation
Уверенность модели неполная; это ограничивает автоматическое доверие.

## Next
risk
