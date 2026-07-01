# Внешняя модель

## Input
- `source_type` = `"image_like_features"`
- `model_name` = `"LogisticRegression"`
- `dataset_name` = `"sklearn_digits"`
- `model_output` = `{"predicted_class": 1, "class_probability": 0.86}`
- `quality_metrics` = `{"missing_rate": 0.0, "feature_range_violation": 0.0}`
- `feature_values` = `{"pixel1": 0.1, "pixel2": 0.2, "pixel3": 0.3, "pixel4": 0.4, "pixel5": 0.5}`
- `attribution_values` = `{"pixel1": 0.2592, "pixel2": 0.1944, "pixel3": 0.1458, "pixel4": 0.1134, "pixel5": 0.0972}`
- `context_values` = `{"experiment_id": "digits_logistic_clean", "task_type": "image_like_classification", "perturbation": "clean", "representation_class": "F_ML", "prediction_interval_width": 0.0, "conflict_component": 0.0, "noise_ratio": 0.0, "occlusion_rate": 0.0}`

## Formula
Нормализация внешнего входа

## Components
- `model_name` = `"LogisticRegression"`
- `dataset_name` = `"sklearn_digits"`
- `source_type` = `"image_like_features"`
- `task_type` = `"image_like_classification"`
- `perturbation` = `"clean"`

## Output
- `adapted_input_id` = `"external_wine_classification:LogisticRegression"`
- `trace_origin` = `"external_model_payload"`
- `class_probability` = `0.86`
- `feature_importance` = `{"pixel1": 0.2592, "pixel2": 0.1944, "pixel3": 0.1458, "pixel4": 0.1134, "pixel5": 0.0972}`

## Thresholds
n/a

## Status
passed: Внешний payload успешно приведён к AdaptedInput.

## Interpretation
Фреймворк получил уверенность модели, признаки, важности признаков, метрики качества и контекст ограничения.

## Next
explanation_object
