# Действие

## Input
- `rho` = `0.62`
- `diagnostic_id` = `"D_external_tabular_quality_limit"`
- `risk_zone` = `"audit"`

## Formula
if rho < 0.35: accept; elif rho < 0.60: lower_confidence; else: audit

## Components
- `rho` = `0.62`
- `diagnostic_id` = `"D_external_tabular_quality_limit"`
- `alternative_actions` = `["accept", "lower_confidence", "audit"]`

## Output
- `action_id` = `"audit"`
- `action_title_ru` = `"audit"`

## Thresholds
- `theta_accept` = `0.35`
- `theta_warning` = `0.6`

## Status
warning: Риск ненулевой, но не критический.

## Interpretation
Результат не блокируется, но автоматическое доверие понижается из-за ненулевого риска.

## Next
proof
