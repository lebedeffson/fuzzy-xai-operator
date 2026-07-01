# Действие

## Input
- `rho` = `0.22`
- `diagnostic_id` = `"D_external_tabular_ok"`
- `risk_zone` = `"accept"`

## Formula
if rho < 0.35: accept; elif rho < 0.60: lower_confidence; else: audit

## Components
- `rho` = `0.22`
- `diagnostic_id` = `"D_external_tabular_ok"`
- `alternative_actions` = `["accept", "lower_confidence", "audit"]`

## Output
- `action_id` = `"accept"`
- `action_title_ru` = `"accept"`

## Thresholds
- `theta_accept` = `0.35`
- `theta_warning` = `0.6`

## Status
passed: Риск ненулевой, но не критический.

## Interpretation
Результат не блокируется, но автоматическое доверие понижается из-за ненулевого риска.

## Next
proof
