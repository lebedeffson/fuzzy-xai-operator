# Диагностика D

## Input
- `rho` = `0.43`
- `risk_zone` = `"lower_confidence"`
- `gamma` = `0.43`
- `delta` = `0.25`
- `task_type` = `"signal_quality"`
- `dominant_component` = `"gamma"`

## Formula
diagnostic_id выбирается по task_type, risk_zone и доминирующему компоненту риска

## Components
- `reason_components` = `{"gamma": 0.43, "delta": 0.25, "rho": 0.43, "dominant_component": "gamma", "task_type": "signal_quality"}`

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
