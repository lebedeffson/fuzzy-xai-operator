# Диагностика D

## Input
- `rho` = `0.58`
- `risk_zone` = `"lower_confidence"`
- `gamma` = `0.58`
- `delta` = `0.37`
- `task_type` = `"tabular_regression"`
- `dominant_component` = `"gamma"`

## Formula
diagnostic_id выбирается по task_type, risk_zone и доминирующему компоненту риска

## Components
- `reason_components` = `{"gamma": 0.58, "delta": 0.37, "rho": 0.58, "dominant_component": "gamma", "task_type": "tabular_regression"}`

## Output
- `diagnostic_id` = `"D_external_regression_uncertainty"`
- `diagnostic_title_ru` = `"ограниченная уверенность внешней регрессионной модели"`
- `severity` = `"medium"`
- `recommended_action_level` = `"lower_confidence"`

## Thresholds
n/a

## Status
warning: ограниченная уверенность внешней регрессионной модели

## Interpretation
Диагностика означает ограничение доверия, а не запрет или утверждение о плохой модели.

## Next
action
