# Внешняя модель

## Input
- `source_type` = `"tabular_regression"`
- `model_name` = `"SVR"`
- `dataset_name` = `"synthetic_regression"`
- `model_output` = `{"predicted_class": 1, "class_probability": 0.71}`
- `quality_metrics` = `{"missing_rate": 0.0, "feature_range_violation": 0.0}`
- `feature_values` = `{"f1": 0.1, "f2": 0.2, "f3": 0.3, "f4": 0.4, "f5": 0.5}`
- `attribution_values` = `{"f1": 0.144, "f2": 0.108, "f3": 0.081, "f4": 0.063, "f5": 0.054}`
- `context_values` = `{"experiment_id": "synthetic_svr_explanation_loss", "task_type": "tabular_regression", "perturbation": "top_k_reduction", "representation_class": "F_int", "prediction_interval_width": 0.3, "conflict_component": 0.0, "noise_ratio": 0.0, "occlusion_rate": 0.0}`

## Formula
Нормализация внешнего входа

## Components
- `model_name` = `"SVR"`
- `dataset_name` = `"synthetic_regression"`
- `source_type` = `"tabular_regression"`
- `task_type` = `"tabular_regression"`
- `perturbation` = `"top_k_reduction"`

## Output
- `adapted_input_id` = `"external_wine_classification:SVR"`
- `trace_origin` = `"external_model_payload"`
- `class_probability` = `0.71`
- `feature_importance` = `{"f1": 0.144, "f2": 0.108, "f3": 0.081, "f4": 0.063, "f5": 0.054}`

## Thresholds
n/a

## Status
passed: Внешний payload успешно приведён к AdaptedInput.

## Interpretation
Фреймворк получил уверенность модели, признаки, важности признаков, метрики качества и контекст ограничения.

## Next
explanation_object
