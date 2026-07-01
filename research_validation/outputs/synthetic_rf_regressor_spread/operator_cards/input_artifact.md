# Внешняя модель

## Input
- `source_type` = `"tabular_regression"`
- `model_name` = `"RandomForestRegressor"`
- `dataset_name` = `"synthetic_regression"`
- `model_output` = `{"predicted_class": 1, "class_probability": 0.55}`
- `quality_metrics` = `{"missing_rate": 0.0, "feature_range_violation": 0.0}`
- `feature_values` = `{"f1": 0.1, "f2": 0.2, "f3": 0.3, "f4": 0.4, "f5": 0.5}`
- `attribution_values` = `{"f1": 0.2016, "f2": 0.1512, "f3": 0.1134, "f4": 0.0882, "f5": 0.0756}`
- `context_values` = `{"experiment_id": "synthetic_rf_regressor_spread", "task_type": "tabular_regression", "perturbation": "wide_interval", "representation_class": "F_int", "prediction_interval_width": 0.58, "conflict_component": 0.0, "noise_ratio": 0.0, "occlusion_rate": 0.0}`

## Formula
Нормализация внешнего входа

## Components
- `model_name` = `"RandomForestRegressor"`
- `dataset_name` = `"synthetic_regression"`
- `source_type` = `"tabular_regression"`
- `task_type` = `"tabular_regression"`
- `perturbation` = `"wide_interval"`

## Output
- `adapted_input_id` = `"external_wine_classification:RandomForestRegressor"`
- `trace_origin` = `"external_model_payload"`
- `class_probability` = `0.55`
- `feature_importance` = `{"f1": 0.2016, "f2": 0.1512, "f3": 0.1134, "f4": 0.0882, "f5": 0.0756}`

## Thresholds
n/a

## Status
passed: Внешний payload успешно приведён к AdaptedInput.

## Interpretation
Фреймворк получил уверенность модели, признаки, важности признаков, метрики качества и контекст ограничения.

## Next
explanation_object
