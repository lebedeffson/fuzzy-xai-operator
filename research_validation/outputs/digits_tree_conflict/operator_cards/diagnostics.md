# Диагностика D

## Input
- `rho` = `0.67`
- `risk_zone` = `"audit"`
- `gamma` = `0.67`
- `delta` = `0.29`
- `task_type` = `"tabular_classification"`
- `dominant_component` = `"gamma"`

## Formula
diagnostic_id выбирается по task_type, risk_zone и доминирующему компоненту риска

## Components
- `reason_components` = `{"gamma": 0.67, "delta": 0.29, "rho": 0.67, "dominant_component": "gamma", "task_type": "tabular_classification"}`

## Output
- `diagnostic_id` = `"D_rule_attribution_conflict"`
- `diagnostic_title_ru` = `"конфликт объяснительного источника и модельного сигнала"`
- `severity` = `"high"`
- `recommended_action_level` = `"audit"`

## Thresholds
n/a

## Status
warning: конфликт объяснительного источника и модельного сигнала

## Interpretation
Диагностика означает ограничение доверия, а не запрет или утверждение о плохой модели.

## Next
action
