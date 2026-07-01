# Диагностика D

## Input
- `rho` = `0.52`
- `risk_zone` = `"lower_confidence"`
- `gamma` = `0.24`
- `delta` = `0.52`
- `task_type` = `"tabular_classification"`
- `dominant_component` = `"delta"`

## Formula
diagnostic_id выбирается по task_type, risk_zone и доминирующему компоненту риска

## Components
- `reason_components` = `{"gamma": 0.24, "delta": 0.52, "rho": 0.52, "dominant_component": "delta", "task_type": "tabular_classification"}`

## Output
- `diagnostic_id` = `"D_external_tabular_reduction_loss"`
- `diagnostic_title_ru` = `"потери редуцированного табличного объяснения"`
- `severity` = `"medium"`
- `recommended_action_level` = `"lower_confidence"`

## Thresholds
n/a

## Status
warning: потери редуцированного табличного объяснения

## Interpretation
Диагностика означает ограничение доверия, а не запрет или утверждение о плохой модели.

## Next
action
