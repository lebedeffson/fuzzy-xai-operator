# Диагностика D

## Input
- `rho` = `0.5`
- `risk_zone` = `"lower_confidence"`
- `gamma` = `0.3`
- `delta` = `0.5`
- `task_type` = `"image_like_classification"`
- `dominant_component` = `"delta"`

## Formula
diagnostic_id выбирается по task_type, risk_zone и доминирующему компоненту риска

## Components
- `reason_components` = `{"gamma": 0.3, "delta": 0.5, "rho": 0.5, "dominant_component": "delta", "task_type": "image_like_classification"}`

## Output
- `diagnostic_id` = `"D_image_explanation_reduction"`
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
