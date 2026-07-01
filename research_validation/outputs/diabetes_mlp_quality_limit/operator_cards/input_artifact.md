# Внешняя модель

## Input
- `source_type` = `"tabular_regression"`
- `model_name` = `"MLPRegressor"`
- `dataset_name` = `"sklearn_diabetes"`
- `model_output` = `{"predicted_class": 1, "class_probability": 0.74}`
- `quality_metrics` = `{"missing_rate": 0.52, "feature_range_violation": 0.0}`
- `feature_values` = `{"f1": 0.1, "f2": 0.2, "f3": 0.3, "f4": 0.4, "f5": 0.5}`
- `attribution_values` = `{"f1": 0.2304, "f2": 0.1728, "f3": 0.1296, "f4": 0.1008, "f5": 0.0864}`
- `context_values` = `{"experiment_id": "diabetes_mlp_quality_limit", "task_type": "tabular_regression", "perturbation": "missing_features", "representation_class": "NAS", "prediction_interval_width": 0.36, "conflict_component": 0.0, "noise_ratio": 0.0, "occlusion_rate": 0.0}`

## Formula
Нормализация внешнего входа

## Components
- `model_name` = `"MLPRegressor"`
- `dataset_name` = `"sklearn_diabetes"`
- `source_type` = `"tabular_regression"`
- `task_type` = `"tabular_regression"`
- `perturbation` = `"missing_features"`

## Output
- `adapted_input_id` = `"external_wine_classification:MLPRegressor"`
- `trace_origin` = `"external_model_payload"`
- `class_probability` = `0.74`
- `feature_importance` = `{"f1": 0.2304, "f2": 0.1728, "f3": 0.1296, "f4": 0.1008, "f5": 0.0864}`

## Thresholds
n/a

## Status
passed: Внешний payload успешно приведён к AdaptedInput.

## Interpretation
Фреймворк получил уверенность модели, признаки, важности признаков, метрики качества и контекст ограничения.

## Next
explanation_object
