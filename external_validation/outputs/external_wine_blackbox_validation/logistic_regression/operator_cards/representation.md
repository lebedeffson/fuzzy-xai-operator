# Класс представления F

## Input
- `source_type` = `"tabular"`
- `task_type` = `"tabular_classification"`
- `has_single_probability` = `true`
- `has_top_k_importance` = `true`
- `context` = `{"external_task": true, "model_key": "logistic_regression", "top_k_importance": 5, "sample_index": 24, "train_size": 115, "test_size": 63}`

## Formula
class_id выбирается по task_type, interval, source/quality conflict и multilevel context

## Components
- `input_conditions` = `["tabular_classification", "external_payload"]`
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
