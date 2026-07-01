# Риск rho

## Input
- `gamma` = `0.61`
- `delta` = `0.28`
- `quality_component` = `0.61`
- `conflict_component` = `0.0`

## Formula
rho = max(gamma, delta, quality_component, conflict_component)

## Components
- `gamma` = `0.61`
- `delta` = `0.28`
- `quality_component` = `0.61`
- `conflict_component` = `0.0`
- `calculation` = `"max(0.61, 0.28, 0.61, 0.0) = 0.61"`

## Output
- `rho` = `0.61`
- `risk_zone` = `"audit"`
- `dominant_component` = `"gamma"`

## Thresholds
- `theta_accept` = `0.35`
- `theta_warning` = `0.6`

## Status
warning: rho попал в зону audit.

## Interpretation
Основной вклад в риск: gamma.

## Next
diagnostics
