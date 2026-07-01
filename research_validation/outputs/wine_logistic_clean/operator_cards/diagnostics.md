# Диагностика D

## Input
- `rho` = `0.18`
- `risk_zone` = `"accept"`
- `gamma` = `0.12`
- `delta` = `0.18`
- `task_type` = `"tabular_classification"`
- `dominant_component` = `"delta"`

## Formula
diagnostic_id выбирается по task_type, risk_zone и доминирующему компоненту риска

## Components
- `reason_components` = `{"gamma": 0.12, "delta": 0.18, "rho": 0.18, "dominant_component": "delta", "task_type": "tabular_classification"}`

## Output
- `diagnostic_id` = `"D_external_tabular_ok"`
- `diagnostic_title_ru` = `"внешний результат допустим"`
- `severity` = `"low"`
- `recommended_action_level` = `"accept"`

## Thresholds
n/a

## Status
passed: внешний результат допустим

## Interpretation
Диагностика означает ограничение доверия, а не запрет или утверждение о плохой модели.

## Next
action
