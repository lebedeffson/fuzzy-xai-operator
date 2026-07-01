# Внешняя модель

## Input
- `source_type` = `"signal_features"`
- `model_name` = `"LogisticRegression"`
- `dataset_name` = `"synthetic_signal"`
- `model_output` = `{"predicted_class": 1, "class_probability": 0.57}`
- `quality_metrics` = `{"missing_rate": 0.0, "feature_range_violation": 0.0}`
- `feature_values` = `{"f1": 0.1, "f2": 0.2, "f3": 0.3, "f4": 0.4, "f5": 0.5}`
- `attribution_values` = `{"f1": 0.24, "f2": 0.18, "f3": 0.135, "f4": 0.105, "f5": 0.09}`
- `context_values` = `{"experiment_id": "signal_logistic_boundary", "task_type": "signal_quality", "perturbation": "confidence_boundary", "representation_class": "F_ML", "prediction_interval_width": 0.0, "conflict_component": 0.0, "noise_ratio": 0.15, "occlusion_rate": 0.0}`

## Formula
Нормализация внешнего входа

## Components
- `model_name` = `"LogisticRegression"`
- `dataset_name` = `"synthetic_signal"`
- `source_type` = `"signal_features"`
- `task_type` = `"signal_quality"`
- `perturbation` = `"confidence_boundary"`

## Output
- `adapted_input_id` = `"external_wine_classification:LogisticRegression"`
- `trace_origin` = `"external_model_payload"`
- `class_probability` = `0.57`
- `feature_importance` = `{"f1": 0.24, "f2": 0.18, "f3": 0.135, "f4": 0.105, "f5": 0.09}`

## Thresholds
n/a

## Status
passed: Внешний payload успешно приведён к AdaptedInput.

## Interpretation
Фреймворк получил уверенность модели, признаки, важности признаков, метрики качества и контекст ограничения.

## Next
explanation_object
