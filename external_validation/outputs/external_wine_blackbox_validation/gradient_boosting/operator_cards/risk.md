# Риск rho

## Input
- `gamma` = `0.320881`
- `delta` = `0.429844`
- `quality_component` = `0.0`
- `conflict_component` = `0.0`

## Formula
rho = max(gamma, delta, quality_component, conflict_component)

## Components
- `gamma` = `0.320881`
- `delta` = `0.429844`
- `quality_component` = `0.0`
- `conflict_component` = `0.0`
- `calculation` = `"max(0.320881, 0.429844, 0.0, 0.0) = 0.429844"`

## Output
- `rho` = `0.429844`
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
