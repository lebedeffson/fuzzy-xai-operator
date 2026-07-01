# Диагностика D

## Input
- `rho` = `0.64`
- `risk_zone` = `"audit"`
- `gamma` = `0.64`
- `delta` = `0.3`
- `task_type` = `"image_like_classification"`
- `dominant_component` = `"gamma"`

## Formula
diagnostic_id выбирается по task_type, risk_zone и доминирующему компоненту риска

## Components
- `reason_components` = `{"gamma": 0.64, "delta": 0.3, "rho": 0.64, "dominant_component": "gamma", "task_type": "image_like_classification"}`

## Output
- `diagnostic_id` = `"D_image_quality_limit"`
- `diagnostic_title_ru` = `"ограничение качества изображения"`
- `severity` = `"high"`
- `recommended_action_level` = `"audit"`

## Thresholds
n/a

## Status
warning: ограничение качества изображения

## Interpretation
Диагностика означает ограничение доверия, а не запрет или утверждение о плохой модели.

## Next
action
