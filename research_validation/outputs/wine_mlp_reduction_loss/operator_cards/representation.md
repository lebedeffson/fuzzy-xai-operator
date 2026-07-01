# Класс представления F

## Input
- `source_type` = `"tabular"`
- `task_type` = `"tabular_classification"`
- `has_single_probability` = `true`
- `has_top_k_importance` = `true`
- `context` = `{"experiment_id": "wine_mlp_reduction_loss", "task_type": "tabular_classification", "perturbation": "top_k_reduction", "representation_class": "F_ML", "prediction_interval_width": 0.0, "conflict_component": 0.0, "noise_ratio": 0.0, "occlusion_rate": 0.0}`

## Formula
class_id выбирается по task_type, interval, source/quality conflict и multilevel context

## Components
- `input_conditions` = `["tabular_classification", "top_k_reduction"]`
- `interval_width` = `0.0`
- `quality_penalty` = `0.0`
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
