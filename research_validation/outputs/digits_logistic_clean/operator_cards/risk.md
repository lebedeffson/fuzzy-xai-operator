# Риск rho

## Input
- `gamma` = `0.14`
- `delta` = `0.19`
- `quality_component` = `0.0`
- `conflict_component` = `0.0`

## Formula
rho = max(gamma, delta, quality_component, conflict_component)

## Components
- `gamma` = `0.14`
- `delta` = `0.19`
- `quality_component` = `0.0`
- `conflict_component` = `0.0`
- `calculation` = `"max(0.14, 0.19, 0.0, 0.0) = 0.19"`

## Output
- `rho` = `0.19`
- `risk_zone` = `"accept"`
- `dominant_component` = `"delta"`

## Thresholds
- `theta_accept` = `0.35`
- `theta_warning` = `0.6`

## Status
passed: rho попал в зону accept.

## Interpretation
Основной вклад в риск: delta.

## Next
diagnostics
