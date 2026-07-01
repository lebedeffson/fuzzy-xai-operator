# Диагностика D

## Input
- `rho` = `0.61`
- `risk_zone` = `"audit"`
- `gamma` = `0.61`
- `delta` = `0.28`
- `task_type` = `"signal_quality"`
- `dominant_component` = `"gamma"`

## Formula
diagnostic_id выбирается по task_type, risk_zone и доминирующему компоненту риска

## Components
- `reason_components` = `{"gamma": 0.61, "delta": 0.28, "rho": 0.61, "dominant_component": "gamma", "task_type": "signal_quality"}`

## Output
- `diagnostic_id` = `"D_signal_noise_limit"`
- `diagnostic_title_ru` = `"ограничение качества сигнала"`
- `severity` = `"high"`
- `recommended_action_level` = `"defer_to_human"`

## Thresholds
n/a

## Status
warning: ограничение качества сигнала

## Interpretation
Диагностика означает ограничение доверия, а не запрет или утверждение о плохой модели.

## Next
action
