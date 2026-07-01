# Риск rho

## Input
- `gamma` = `0.3`
- `delta` = `0.5`
- `quality_component` = `0.12`
- `conflict_component` = `0.0`

## Formula
rho = max(gamma, delta, quality_component, conflict_component)

## Components
- `gamma` = `0.3`
- `delta` = `0.5`
- `quality_component` = `0.12`
- `conflict_component` = `0.0`
- `calculation` = `"max(0.3, 0.5, 0.12, 0.0) = 0.5"`

## Output
- `rho` = `0.5`
- `risk_zone` = `"lower_confidence"`
- `dominant_component` = `"delta"`

## Thresholds
- `theta_accept` = `0.35`
- `theta_warning` = `0.6`

## Status
warning: rho попал в зону lower_confidence.

## Interpretation
Основной вклад в риск: delta.

## Next
diagnostics
