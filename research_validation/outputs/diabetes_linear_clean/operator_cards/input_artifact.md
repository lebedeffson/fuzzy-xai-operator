# Внешняя модель

## Input
- `source_type` = `"tabular_regression"`
- `model_name` = `"LinearRegression"`
- `dataset_name` = `"sklearn_diabetes"`
- `model_output` = `{"predicted_class": 1, "class_probability": 0.82}`
- `quality_metrics` = `{"missing_rate": 0.0, "feature_range_violation": 0.0}`
- `feature_values` = `{"f1": 0.1, "f2": 0.2, "f3": 0.3, "f4": 0.4, "f5": 0.5}`
- `attribution_values` = `{"f1": 0.256, "f2": 0.192, "f3": 0.144, "f4": 0.112, "f5": 0.096}`
- `context_values` = `{"experiment_id": "diabetes_linear_clean", "task_type": "tabular_regression", "perturbation": "clean", "representation_class": "F_int", "prediction_interval_width": 0.22, "conflict_component": 0.0, "noise_ratio": 0.0, "occlusion_rate": 0.0}`

## Formula
Нормализация внешнего входа

## Components
- `model_name` = `"LinearRegression"`
- `dataset_name` = `"sklearn_diabetes"`
- `source_type` = `"tabular_regression"`
- `task_type` = `"tabular_regression"`
- `perturbation` = `"clean"`

## Output
- `adapted_input_id` = `"external_wine_classification:LinearRegression"`
- `trace_origin` = `"external_model_payload"`
- `class_probability` = `0.82`
- `feature_importance` = `{"f1": 0.256, "f2": 0.192, "f3": 0.144, "f4": 0.112, "f5": 0.096}`

## Thresholds
n/a

## Status
passed: Внешний payload успешно приведён к AdaptedInput.

## Interpretation
Фреймворк получил уверенность модели, признаки, важности признаков, метрики качества и контекст ограничения.

## Next
explanation_object
