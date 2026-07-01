# Диагностика D

## Input
- `rho` = `0.55`
- `risk_zone` = `"lower_confidence"`
- `gamma` = `0.3`
- `delta` = `0.55`
- `task_type` = `"tabular_regression"`
- `dominant_component` = `"delta"`

## Formula
diagnostic_id выбирается по task_type, risk_zone и доминирующему компоненту риска

## Components
- `reason_components` = `{"gamma": 0.3, "delta": 0.55, "rho": 0.55, "dominant_component": "delta", "task_type": "tabular_regression"}`

## Output
- `diagnostic_id` = `"D_external_regression_explanation_loss"`
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
