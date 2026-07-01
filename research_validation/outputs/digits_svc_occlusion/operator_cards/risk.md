# Риск rho

## Input
- `gamma` = `0.64`
- `delta` = `0.3`
- `quality_component` = `0.64`
- `conflict_component` = `0.0`

## Formula
rho = max(gamma, delta, quality_component, conflict_component)

## Components
- `gamma` = `0.64`
- `delta` = `0.3`
- `quality_component` = `0.64`
- `conflict_component` = `0.0`
- `calculation` = `"max(0.64, 0.3, 0.64, 0.0) = 0.64"`

## Output
- `rho` = `0.64`
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
