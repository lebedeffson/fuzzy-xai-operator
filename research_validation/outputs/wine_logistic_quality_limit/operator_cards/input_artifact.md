# Внешняя модель

## Input
- `source_type` = `"tabular"`
- `model_name` = `"LogisticRegression"`
- `dataset_name` = `"sklearn_wine"`
- `model_output` = `{"predicted_class": 1, "class_probability": 0.84}`
- `quality_metrics` = `{"missing_rate": 0.41, "feature_range_violation": 0.2}`
- `feature_values` = `{"f1": 0.1, "f2": 0.2, "f3": 0.3, "f4": 0.4, "f5": 0.5}`
- `attribution_values` = `{"f1": 0.2432, "f2": 0.1824, "f3": 0.1368, "f4": 0.1064, "f5": 0.0912}`
- `context_values` = `{"experiment_id": "wine_logistic_quality_limit", "task_type": "tabular_classification", "perturbation": "missing_features", "representation_class": "NAS", "prediction_interval_width": 0.0, "conflict_component": 0.0, "noise_ratio": 0.0, "occlusion_rate": 0.0}`

## Formula
Нормализация внешнего входа

## Components
- `model_name` = `"LogisticRegression"`
- `dataset_name` = `"sklearn_wine"`
- `source_type` = `"tabular"`
- `task_type` = `"tabular_classification"`
- `perturbation` = `"missing_features"`

## Output
- `adapted_input_id` = `"external_wine_classification:LogisticRegression"`
- `trace_origin` = `"external_model_payload"`
- `class_probability` = `0.84`
- `feature_importance` = `{"f1": 0.2432, "f2": 0.1824, "f3": 0.1368, "f4": 0.1064, "f5": 0.0912}`

## Thresholds
n/a

## Status
passed: Внешний payload успешно приведён к AdaptedInput.

## Interpretation
Фреймворк получил уверенность модели, признаки, важности признаков, метрики качества и контекст ограничения.

## Next
explanation_object
