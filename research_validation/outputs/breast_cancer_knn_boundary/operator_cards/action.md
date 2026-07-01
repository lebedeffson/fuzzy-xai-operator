# Действие

## Input
- `rho` = `0.42`
- `diagnostic_id` = `"D_external_tabular_uncertainty"`
- `risk_zone` = `"lower_confidence"`

## Formula
if rho < 0.35: accept; elif rho < 0.60: lower_confidence; else: audit

## Components
- `rho` = `0.42`
- `diagnostic_id` = `"D_external_tabular_uncertainty"`
- `alternative_actions` = `["accept", "lower_confidence", "audit"]`

## Output
- `action_id` = `"lower_confidence"`
- `action_title_ru` = `"понизить доверие"`

## Thresholds
- `theta_accept` = `0.35`
- `theta_warning` = `0.6`

## Status
warning: Риск ненулевой, но не критический.

## Interpretation
Результат не блокируется, но автоматическое доверие понижается из-за ненулевого риска.

## Next
proof
