# Диагностика D

## Input
- `legacy_route_score` = `0.373128`
- `risk_zone` = `"lower_confidence"`
- `legacy_route_gap` = `0.310276`
- `presentation_omission_loss` = `0.373128`
- `task_type` = `"tabular_classification"`
- `dominant_component` = `"presentation_omission"`

## Formula
diagnostic_id выбирается по task_type, risk_zone и доминирующему компоненту риска

## Components
- `reason_components` = `{"legacy_route_gap": 0.310276, "presentation_omission_loss": 0.373128, "legacy_route_score": 0.373128, "dominant_component": "presentation_omission", "task_type": "tabular_classification"}`

## Output
- `diagnostic_id` = `"D_external_tabular_uncertainty"`
- `diagnostic_title_ru` = `"ограниченная уверенность внешней табличной модели"`
- `severity` = `"medium"`
- `recommended_action_level` = `"lower_confidence"`

## Thresholds
n/a

## Status
warning: ограниченная уверенность внешней табличной модели

## Interpretation
Диагностика означает ограничение доверия, а не запрет или утверждение о плохой модели.

## Next
action
