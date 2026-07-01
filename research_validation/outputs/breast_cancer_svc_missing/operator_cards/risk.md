# Риск rho

## Input
- `gamma` = `0.62`
- `delta` = `0.3`
- `quality_component` = `0.62`
- `conflict_component` = `0.0`

## Formula
rho = max(gamma, delta, quality_component, conflict_component)

## Components
- `gamma` = `0.62`
- `delta` = `0.3`
- `quality_component` = `0.62`
- `conflict_component` = `0.0`
- `calculation` = `"max(0.62, 0.3, 0.62, 0.0) = 0.62"`

## Output
- `rho` = `0.62`
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
