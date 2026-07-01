# Класс представления F

## Input
- `source_type` = `"signal_features"`
- `task_type` = `"signal_quality"`
- `has_single_probability` = `true`
- `has_top_k_importance` = `true`
- `context` = `{"experiment_id": "signal_gradient_missing", "task_type": "signal_quality", "perturbation": "missing_features", "representation_class": "F_ML", "prediction_interval_width": 0.0, "conflict_component": 0.0, "noise_ratio": 0.35, "occlusion_rate": 0.0}`

## Formula
class_id выбирается по task_type, interval, source/quality conflict и multilevel context

## Components
- `input_conditions` = `["signal_quality", "missing_features"]`
- `interval_width` = `0.0`
- `quality_penalty` = `0.73`
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
