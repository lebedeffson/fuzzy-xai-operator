# Внешняя модель

## Input
- `source_type` = `"tabular"`
- `model_name` = `"GradientBoostingClassifier"`
- `dataset_name` = `"sklearn_wine"`
- `model_output` = `{"predicted_class": 1, "class_probability": 0.68}`
- `quality_metrics` = `{"missing_rate": 0.0, "feature_range_violation": 0.0}`
- `feature_values` = `{"f1": 0.1, "f2": 0.2, "f3": 0.3, "f4": 0.4, "f5": 0.5}`
- `attribution_values` = `{"f1": 0.1824, "f2": 0.1368, "f3": 0.1026, "f4": 0.0798, "f5": 0.0684}`
- `context_values` = `{"experiment_id": "wine_gradient_reduced", "task_type": "tabular_classification", "perturbation": "top_k_reduction", "representation_class": "F0", "prediction_interval_width": 0.0, "conflict_component": 0.0, "noise_ratio": 0.0, "occlusion_rate": 0.0}`

## Formula
Нормализация внешнего входа

## Components
- `model_name` = `"GradientBoostingClassifier"`
- `dataset_name` = `"sklearn_wine"`
- `source_type` = `"tabular"`
- `task_type` = `"tabular_classification"`
- `perturbation` = `"top_k_reduction"`

## Output
- `adapted_input_id` = `"external_wine_classification:GradientBoostingClassifier"`
- `trace_origin` = `"external_model_payload"`
- `class_probability` = `0.68`
- `feature_importance` = `{"f1": 0.1824, "f2": 0.1368, "f3": 0.1026, "f4": 0.0798, "f5": 0.0684}`

## Thresholds
n/a

## Status
passed: Внешний payload успешно приведён к AdaptedInput.

## Interpretation
Фреймворк получил уверенность модели, признаки, важности признаков, метрики качества и контекст ограничения.

## Next
explanation_object
