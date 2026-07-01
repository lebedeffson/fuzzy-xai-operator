# Внешняя модель

## Input
- `source_type` = `"signal_features"`
- `model_name` = `"GradientBoostingClassifier"`
- `dataset_name` = `"synthetic_signal"`
- `model_output` = `{"predicted_class": 1, "class_probability": 0.8}`
- `quality_metrics` = `{"missing_rate": 0.73, "feature_range_violation": 0.0}`
- `feature_values` = `{"f1": 0.1, "f2": 0.2, "f3": 0.3, "f4": 0.4, "f5": 0.5}`
- `attribution_values` = `{"f1": 0.2176, "f2": 0.1632, "f3": 0.1224, "f4": 0.0952, "f5": 0.0816}`
- `context_values` = `{"experiment_id": "signal_gradient_missing", "task_type": "signal_quality", "perturbation": "missing_features", "representation_class": "F_ML", "prediction_interval_width": 0.0, "conflict_component": 0.0, "noise_ratio": 0.35, "occlusion_rate": 0.0}`

## Formula
Нормализация внешнего входа

## Components
- `model_name` = `"GradientBoostingClassifier"`
- `dataset_name` = `"synthetic_signal"`
- `source_type` = `"signal_features"`
- `task_type` = `"signal_quality"`
- `perturbation` = `"missing_features"`

## Output
- `adapted_input_id` = `"external_wine_classification:GradientBoostingClassifier"`
- `trace_origin` = `"external_model_payload"`
- `class_probability` = `0.8`
- `feature_importance` = `{"f1": 0.2176, "f2": 0.1632, "f3": 0.1224, "f4": 0.0952, "f5": 0.0816}`

## Thresholds
n/a

## Status
passed: Внешний payload успешно приведён к AdaptedInput.

## Interpretation
Фреймворк получил уверенность модели, признаки, важности признаков, метрики качества и контекст ограничения.

## Next
explanation_object
