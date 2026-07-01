# Класс представления F

## Input
- `source_type` = `"image_like_features"`
- `task_type` = `"image_like_classification"`
- `has_single_probability` = `true`
- `has_top_k_importance` = `true`
- `context` = `{"experiment_id": "digits_gradient_reduction", "task_type": "image_like_classification", "perturbation": "top_k_reduction", "representation_class": "F_ML", "prediction_interval_width": 0.0, "conflict_component": 0.0, "noise_ratio": 0.0, "occlusion_rate": 0.12}`

## Formula
class_id выбирается по task_type, interval, source/quality conflict и multilevel context

## Components
- `input_conditions` = `["image_like_classification", "top_k_reduction"]`
- `interval_width` = `0.0`
- `quality_penalty` = `0.12`
- `conflict_component` = `0.0`

## Output
- `class_id` = `"F_ML"`
- `class_title_ru` = `"многоуровневое представление"`
- `output_representation` = `"confidence + top-k attribution + context metrics"`

## Thresholds
n/a

## Status
passed: Выбран класс F_ML.

## Interpretation
Многоуровневое представление связывает качество входа, признаки, уверенность модели и объяснение.

## Next
alignment, reduction
