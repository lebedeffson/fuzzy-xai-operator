# Риск rho

## Input
- `gamma` = `0.22`
- `delta` = `0.2`
- `quality_component` = `0.0`
- `conflict_component` = `0.0`

## Formula
rho = max(gamma, delta, quality_component, conflict_component)

## Components
- `gamma` = `0.22`
- `delta` = `0.2`
- `quality_component` = `0.0`
- `conflict_component` = `0.0`
- `calculation` = `"max(0.22, 0.2, 0.0, 0.0) = 0.22"`

## Output
- `rho` = `0.22`
- `risk_zone` = `"accept"`
- `dominant_component` = `"gamma"`

## Thresholds
- `theta_accept` = `0.35`
- `theta_warning` = `0.6`

## Status
passed: rho попал в зону accept.

## Interpretation
Основной вклад в риск: gamma.

## Next
diagnostics
