# Внешняя модель

## Input
- `source_type` = `"image_like_features"`
- `model_name` = `"GradientBoostingClassifier"`
- `dataset_name` = `"sklearn_digits"`
- `model_output` = `{"predicted_class": 1, "class_probability": 0.7}`
- `quality_metrics` = `{"missing_rate": 0.0, "feature_range_violation": 0.0}`
- `feature_values` = `{"pixel1": 0.1, "pixel2": 0.2, "pixel3": 0.3, "pixel4": 0.4, "pixel5": 0.5}`
- `attribution_values` = `{"pixel1": 0.16, "pixel2": 0.12, "pixel3": 0.09, "pixel4": 0.07, "pixel5": 0.06}`
- `context_values` = `{"experiment_id": "digits_gradient_reduction", "task_type": "image_like_classification", "perturbation": "top_k_reduction", "representation_class": "F_ML", "prediction_interval_width": 0.0, "conflict_component": 0.0, "noise_ratio": 0.0, "occlusion_rate": 0.12}`

## Formula
Нормализация внешнего входа

## Components
- `model_name` = `"GradientBoostingClassifier"`
- `dataset_name` = `"sklearn_digits"`
- `source_type` = `"image_like_features"`
- `task_type` = `"image_like_classification"`
- `perturbation` = `"top_k_reduction"`

## Output
- `adapted_input_id` = `"external_wine_classification:GradientBoostingClassifier"`
- `trace_origin` = `"external_model_payload"`
- `class_probability` = `0.7`
- `feature_importance` = `{"pixel1": 0.16, "pixel2": 0.12, "pixel3": 0.09, "pixel4": 0.07, "pixel5": 0.06}`

## Thresholds
n/a

## Status
passed: Внешний payload успешно приведён к AdaptedInput.

## Interpretation
Фреймворк получил уверенность модели, признаки, важности признаков, метрики качества и контекст ограничения.

## Next
explanation_object
