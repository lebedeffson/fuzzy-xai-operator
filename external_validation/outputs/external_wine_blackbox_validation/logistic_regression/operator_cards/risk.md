# Риск rho

## Input
- `gamma` = `0.310276`
- `delta` = `0.373128`

## Formula
rho = max(gamma, delta)

## Components
- `gamma` = `0.310276`
- `delta` = `0.373128`
- `calculation` = `"max(0.310276, 0.373128) = 0.373128"`

## Output
- `rho` = `0.373128`
- `risk_zone` = `"lower_confidence"`
- `dominant_component` = `"delta"`

## Thresholds
- `theta_accept` = `0.35`
- `theta_warning` = `0.6`

## Status
warning: rho попал в зону lower_confidence.

## Interpretation
Основной вклад в риск дала потеря редуцированного объяснения.

## Next
diagnostics
