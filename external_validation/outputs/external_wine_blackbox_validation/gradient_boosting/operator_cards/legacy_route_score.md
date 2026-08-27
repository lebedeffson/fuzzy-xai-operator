# Legacy route score (не rho)

## Input
- `legacy_route_gap` = `0.320881`
- `presentation_omission_loss` = `0.429844`
- `quality_component` = `0.0`
- `conflict_component` = `0.0`

## Formula
legacy_route_score = max(route_gap, presentation_omission, quality, conflict)

## Components
- `legacy_route_gap` = `0.320881`
- `presentation_omission_loss` = `0.429844`
- `quality_component` = `0.0`
- `conflict_component` = `0.0`
- `calculation` = `"max(0.320881, 0.429844, 0.0, 0.0) = 0.429844"`

## Output
- `legacy_route_score` = `0.429844`
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
