# Внешняя модель

## Input
- `source_type` = `"image_like_features"`
- `model_name` = `"MLPClassifier"`
- `dataset_name` = `"sklearn_digits"`
- `model_output` = `{"predicted_class": 1, "class_probability": 0.66}`
- `quality_metrics` = `{"missing_rate": 0.0, "feature_range_violation": 0.0}`
- `feature_values` = `{"pixel1": 0.1, "pixel2": 0.2, "pixel3": 0.3, "pixel4": 0.4, "pixel5": 0.5}`
- `attribution_values` = `{"pixel1": 0.1888, "pixel2": 0.1416, "pixel3": 0.1062, "pixel4": 0.0826, "pixel5": 0.0708}`
- `context_values` = `{"experiment_id": "digits_mlp_pixel_noise", "task_type": "image_like_classification", "perturbation": "noise", "representation_class": "F_ML", "prediction_interval_width": 0.0, "conflict_component": 0.0, "noise_ratio": 0.0, "occlusion_rate": 0.42}`

## Formula
Нормализация внешнего входа

## Components
- `model_name` = `"MLPClassifier"`
- `dataset_name` = `"sklearn_digits"`
- `source_type` = `"image_like_features"`
- `task_type` = `"image_like_classification"`
- `perturbation` = `"noise"`

## Output
- `adapted_input_id` = `"external_wine_classification:MLPClassifier"`
- `trace_origin` = `"external_model_payload"`
- `class_probability` = `0.66`
- `feature_importance` = `{"pixel1": 0.1888, "pixel2": 0.1416, "pixel3": 0.1062, "pixel4": 0.0826, "pixel5": 0.0708}`

## Thresholds
n/a

## Status
passed: Внешний payload успешно приведён к AdaptedInput.

## Interpretation
Фреймворк получил уверенность модели, признаки, важности признаков, метрики качества и контекст ограничения.

## Next
explanation_object
