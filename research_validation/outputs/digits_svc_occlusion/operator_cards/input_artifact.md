# Внешняя модель

## Input
- `source_type` = `"image_like_features"`
- `model_name` = `"SVC(probability=True)"`
- `dataset_name` = `"sklearn_digits"`
- `model_output` = `{"predicted_class": 1, "class_probability": 0.79}`
- `quality_metrics` = `{"missing_rate": 0.0, "feature_range_violation": 0.0}`
- `feature_values` = `{"pixel1": 0.1, "pixel2": 0.2, "pixel3": 0.3, "pixel4": 0.4, "pixel5": 0.5}`
- `attribution_values` = `{"pixel1": 0.224, "pixel2": 0.168, "pixel3": 0.126, "pixel4": 0.098, "pixel5": 0.084}`
- `context_values` = `{"experiment_id": "digits_svc_occlusion", "task_type": "image_like_classification", "perturbation": "occlusion", "representation_class": "F_ML", "prediction_interval_width": 0.0, "conflict_component": 0.0, "noise_ratio": 0.0, "occlusion_rate": 0.64}`

## Formula
Нормализация внешнего входа

## Components
- `model_name` = `"SVC(probability=True)"`
- `dataset_name` = `"sklearn_digits"`
- `source_type` = `"image_like_features"`
- `task_type` = `"image_like_classification"`
- `perturbation` = `"occlusion"`

## Output
- `adapted_input_id` = `"external_wine_classification:SVC(probability=True)"`
- `trace_origin` = `"external_model_payload"`
- `class_probability` = `0.79`
- `feature_importance` = `{"pixel1": 0.224, "pixel2": 0.168, "pixel3": 0.126, "pixel4": 0.098, "pixel5": 0.084}`

## Thresholds
n/a

## Status
passed: Внешний payload успешно приведён к AdaptedInput.

## Interpretation
Фреймворк получил уверенность модели, признаки, важности признаков, метрики качества и контекст ограничения.

## Next
explanation_object
