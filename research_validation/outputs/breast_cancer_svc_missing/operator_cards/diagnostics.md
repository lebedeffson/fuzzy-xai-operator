# Диагностика D

## Input
- `rho` = `0.62`
- `risk_zone` = `"audit"`
- `gamma` = `0.62`
- `delta` = `0.3`
- `task_type` = `"tabular_classification"`
- `dominant_component` = `"gamma"`

## Formula
diagnostic_id выбирается по task_type, risk_zone и доминирующему компоненту риска

## Components
- `reason_components` = `{"gamma": 0.62, "delta": 0.3, "rho": 0.62, "dominant_component": "gamma", "task_type": "tabular_classification"}`

## Output
- `diagnostic_id` = `"D_external_tabular_quality_limit"`
- `diagnostic_title_ru` = `"ограничение качества внешнего табличного входа"`
- `severity` = `"high"`
- `recommended_action_level` = `"audit"`

## Thresholds
n/a

## Status
warning: ограничение качества внешнего табличного входа

## Interpretation
Диагностика означает ограничение доверия, а не запрет или утверждение о плохой модели.

## Next
action
