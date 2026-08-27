# Legacy route score (не rho)

## Input
- `legacy_route_gap` = `0.310276`
- `presentation_omission_loss` = `0.373128`
- `quality_component` = `0.0`
- `conflict_component` = `0.0`

## Formula
legacy_route_score = max(route_gap, presentation_omission, quality, conflict)

## Components
- `legacy_route_gap` = `0.310276`
- `presentation_omission_loss` = `0.373128`
- `quality_component` = `0.0`
- `conflict_component` = `0.0`
- `calculation` = `"max(0.310276, 0.373128, 0.0, 0.0) = 0.373128"`

## Output
- `legacy_route_score` = `0.373128`
- `risk_zone` = `"lower_confidence"`
- `dominant_component` = `"presentation_omission"`

## Thresholds
- `theta_accept` = `0.35`
- `theta_warning` = `0.6`

## Status
warning: Legacy route score попал в зону lower_confidence.

## Interpretation
Основной вклад в legacy score: presentation_omission.

## Next
diagnostics
