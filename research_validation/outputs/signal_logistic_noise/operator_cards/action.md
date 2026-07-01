# Действие

## Input
- `rho` = `0.61`
- `diagnostic_id` = `"D_signal_noise_limit"`
- `risk_zone` = `"audit"`

## Formula
if rho < 0.35: accept; elif rho < 0.60: lower_confidence; else: audit

## Components
- `rho` = `0.61`
- `diagnostic_id` = `"D_signal_noise_limit"`
- `alternative_actions` = `["accept", "lower_confidence", "audit"]`

## Output
- `action_id` = `"defer_to_human"`
- `action_title_ru` = `"defer_to_human"`

## Thresholds
- `theta_accept` = `0.35`
- `theta_warning` = `0.6`

## Status
warning: Риск ненулевой, но не критический.

## Interpretation
Результат не блокируется, но автоматическое доверие понижается из-за ненулевого риска.

## Next
proof
