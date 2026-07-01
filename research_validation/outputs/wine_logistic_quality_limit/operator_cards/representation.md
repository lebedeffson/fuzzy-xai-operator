# Класс представления F

## Input
- `source_type` = `"tabular"`
- `task_type` = `"tabular_classification"`
- `has_single_probability` = `true`
- `has_top_k_importance` = `true`
- `context` = `{"experiment_id": "wine_logistic_quality_limit", "task_type": "tabular_classification", "perturbation": "missing_features", "representation_class": "NAS", "prediction_interval_width": 0.0, "conflict_component": 0.0, "noise_ratio": 0.0, "occlusion_rate": 0.0}`

## Formula
class_id выбирается по task_type, interval, source/quality conflict и multilevel context

## Components
- `input_conditions` = `["tabular_classification", "missing_features"]`
- `interval_width` = `0.0`
- `quality_penalty` = `0.41`
- `conflict_component` = `0.0`

## Output
- `class_id` = `"NAS"`
- `class_title_ru` = `"источниковое представление"`
- `output_representation` = `"confidence + top-k attribution + context metrics"`

## Thresholds
n/a

## Status
passed: Выбран класс NAS.

## Interpretation
Источниковое представление фиксирует конфликт качества, источника или объяснения.

## Next
alignment, reduction
