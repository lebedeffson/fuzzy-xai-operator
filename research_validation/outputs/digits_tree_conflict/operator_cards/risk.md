# Риск rho

## Input
- `gamma` = `0.67`
- `delta` = `0.29`
- `quality_component` = `0.0`
- `conflict_component` = `0.67`

## Formula
rho = max(gamma, delta, quality_component, conflict_component)

## Components
- `gamma` = `0.67`
- `delta` = `0.29`
- `quality_component` = `0.0`
- `conflict_component` = `0.67`
- `calculation` = `"max(0.67, 0.29, 0.0, 0.67) = 0.67"`

## Output
- `rho` = `0.67`
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
