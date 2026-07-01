# Согласование T_ij

## Input
- `class_probability` = `0.689724`
- `missing_rate` = `0.0`
- `feature_range_violation` = `0.0`

## Formula
gamma = max(1 - class_probability, max(missing_rate, feature_range_violation))

## Components
- `uncertainty` = `0.310276`
- `quality_penalty` = `0.0`
- `calculation` = `"max(0.310276, 0.0) = 0.310276"`

## Output
- `gamma` = `0.310276`

## Thresholds
- `gamma_warning` = `0.35`

## Status
passed: Рассогласование ненулевое, потому что вероятность класса меньше 1.

## Interpretation
Уверенность модели неполная; это ограничивает автоматическое доверие.

## Next
risk
