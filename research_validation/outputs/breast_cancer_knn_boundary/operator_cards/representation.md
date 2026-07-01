# Класс представления F

## Input
- `source_type` = `"tabular"`
- `task_type` = `"tabular_classification"`
- `has_single_probability` = `true`
- `has_top_k_importance` = `true`
- `context` = `{"experiment_id": "breast_cancer_knn_boundary", "task_type": "tabular_classification", "perturbation": "confidence_boundary", "representation_class": "F0", "prediction_interval_width": 0.0, "conflict_component": 0.0, "noise_ratio": 0.0, "occlusion_rate": 0.0}`

## Formula
class_id выбирается по task_type, interval, source/quality conflict и multilevel context

## Components
- `input_conditions` = `["tabular_classification", "confidence_boundary"]`
- `interval_width` = `0.0`
- `quality_penalty` = `0.0`
- `conflict_component` = `0.0`

## Output
- `class_id` = `"F0"`
- `class_title_ru` = `"обычное нечёткое представление"`
- `output_representation` = `"confidence + top-k attribution + context metrics"`

## Thresholds
n/a

## Status
passed: Выбран класс F0.

## Interpretation
Базовое представление достаточно для одной уверенности и top-k объяснения.

## Next
alignment, reduction
