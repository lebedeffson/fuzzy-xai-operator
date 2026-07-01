# Риск rho

## Input
- `gamma` = `0.3`
- `delta` = `0.55`
- `quality_component` = `0.0`
- `conflict_component` = `0.0`

## Formula
rho = max(gamma, delta, quality_component, conflict_component)

## Components
- `gamma` = `0.3`
- `delta` = `0.55`
- `quality_component` = `0.0`
- `conflict_component` = `0.0`
- `calculation` = `"max(0.3, 0.55, 0.0, 0.0) = 0.55"`

## Output
- `rho` = `0.55`
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
