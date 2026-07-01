# Диагностика D

## Input
- `rho` = `0.42`
- `risk_zone` = `"lower_confidence"`
- `gamma` = `0.42`
- `delta` = `0.41`
- `task_type` = `"image_like_classification"`
- `dominant_component` = `"gamma"`

## Formula
diagnostic_id выбирается по task_type, risk_zone и доминирующему компоненту риска

## Components
- `reason_components` = `{"gamma": 0.42, "delta": 0.41, "rho": 0.42, "dominant_component": "gamma", "task_type": "image_like_classification"}`

## Output
- `diagnostic_id` = `"D_external_image_uncertainty"`
- `diagnostic_title_ru` = `"ограничение image-like объяснения"`
- `severity` = `"medium"`
- `recommended_action_level` = `"lower_confidence"`

## Thresholds
n/a

## Status
warning: ограничение image-like объяснения

## Interpretation
Диагностика означает ограничение доверия, а не запрет или утверждение о плохой модели.

## Next
action
