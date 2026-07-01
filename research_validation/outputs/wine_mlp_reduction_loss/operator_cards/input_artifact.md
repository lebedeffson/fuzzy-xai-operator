# Внешняя модель

## Input
- `source_type` = `"tabular"`
- `model_name` = `"MLPClassifier"`
- `dataset_name` = `"sklearn_wine"`
- `model_output` = `{"predicted_class": 1, "class_probability": 0.76}`
- `quality_metrics` = `{"missing_rate": 0.0, "feature_range_violation": 0.0}`
- `feature_values` = `{"f1": 0.1, "f2": 0.2, "f3": 0.3, "f4": 0.4, "f5": 0.5}`
- `attribution_values` = `{"f1": 0.1536, "f2": 0.1152, "f3": 0.0864, "f4": 0.0672, "f5": 0.0576}`
- `context_values` = `{"experiment_id": "wine_mlp_reduction_loss", "task_type": "tabular_classification", "perturbation": "top_k_reduction", "representation_class": "F_ML", "prediction_interval_width": 0.0, "conflict_component": 0.0, "noise_ratio": 0.0, "occlusion_rate": 0.0}`

## Formula
Нормализация внешнего входа

## Components
- `model_name` = `"MLPClassifier"`
- `dataset_name` = `"sklearn_wine"`
- `source_type` = `"tabular"`
- `task_type` = `"tabular_classification"`
- `perturbation` = `"top_k_reduction"`

## Output
- `adapted_input_id` = `"external_wine_classification:MLPClassifier"`
- `trace_origin` = `"external_model_payload"`
- `class_probability` = `0.76`
- `feature_importance` = `{"f1": 0.1536, "f2": 0.1152, "f3": 0.0864, "f4": 0.0672, "f5": 0.0576}`

## Thresholds
n/a

## Status
passed: Внешний payload успешно приведён к AdaptedInput.

## Interpretation
Фреймворк получил уверенность модели, признаки, важности признаков, метрики качества и контекст ограничения.

## Next
explanation_object
