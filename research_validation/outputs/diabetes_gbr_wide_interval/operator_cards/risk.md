# Риск rho

## Input
- `gamma` = `0.48`
- `delta` = `0.3`
- `quality_component` = `0.0`
- `conflict_component` = `0.0`

## Formula
rho = max(gamma, delta, quality_component, conflict_component)

## Components
- `gamma` = `0.48`
- `delta` = `0.3`
- `quality_component` = `0.0`
- `conflict_component` = `0.0`
- `calculation` = `"max(0.48, 0.3, 0.0, 0.0) = 0.48"`

## Output
- `rho` = `0.48`
- `risk_zone` = `"lower_confidence"`
- `dominant_component` = `"gamma"`

## Thresholds
- `theta_accept` = `0.35`
- `theta_warning` = `0.6`

## Status
warning: rho попал в зону lower_confidence.

## Interpretation
Основной вклад в риск: gamma.

## Next
diagnostics
