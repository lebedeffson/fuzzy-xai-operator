# Класс представления F

## Input
- `source_type` = `"tabular_regression"`
- `task_type` = `"tabular_regression"`
- `has_single_probability` = `true`
- `has_top_k_importance` = `true`
- `context` = `{"experiment_id": "synthetic_svr_explanation_loss", "task_type": "tabular_regression", "perturbation": "top_k_reduction", "representation_class": "F_int", "prediction_interval_width": 0.3, "conflict_component": 0.0, "noise_ratio": 0.0, "occlusion_rate": 0.0}`

## Formula
class_id выбирается по task_type, interval, source/quality conflict и multilevel context

## Components
- `input_conditions` = `["tabular_regression", "top_k_reduction"]`
- `interval_width` = `0.3`
- `quality_penalty` = `0.0`
- `conflict_component` = `0.0`

## Output
- `class_id` = `"F_int"`
- `class_title_ru` = `"интервальное нечёткое представление"`
- `output_representation` = `"confidence + top-k attribution + context metrics"`

## Thresholds
n/a

## Status
passed: Выбран класс F_int.

## Interpretation
Интервальное представление фиксирует ширину прогноза или неопределённость регрессии.

## Next
alignment, reduction
