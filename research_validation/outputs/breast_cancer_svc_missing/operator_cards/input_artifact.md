# Внешняя модель

## Input
- `source_type` = `"tabular"`
- `model_name` = `"SVC(probability=True)"`
- `dataset_name` = `"sklearn_breast_cancer"`
- `model_output` = `{"predicted_class": 1, "class_probability": 0.72}`
- `quality_metrics` = `{"missing_rate": 0.62, "feature_range_violation": 0.1}`
- `feature_values` = `{"f1": 0.1, "f2": 0.2, "f3": 0.3, "f4": 0.4, "f5": 0.5}`
- `attribution_values` = `{"f1": 0.224, "f2": 0.168, "f3": 0.126, "f4": 0.098, "f5": 0.084}`
- `context_values` = `{"experiment_id": "breast_cancer_svc_missing", "task_type": "tabular_classification", "perturbation": "missing_features", "representation_class": "NAS", "prediction_interval_width": 0.0, "conflict_component": 0.0, "noise_ratio": 0.0, "occlusion_rate": 0.0}`

## Formula
Нормализация внешнего входа

## Components
- `model_name` = `"SVC(probability=True)"`
- `dataset_name` = `"sklearn_breast_cancer"`
- `source_type` = `"tabular"`
- `task_type` = `"tabular_classification"`
- `perturbation` = `"missing_features"`

## Output
- `adapted_input_id` = `"external_wine_classification:SVC(probability=True)"`
- `trace_origin` = `"external_model_payload"`
- `class_probability` = `0.72`
- `feature_importance` = `{"f1": 0.224, "f2": 0.168, "f3": 0.126, "f4": 0.098, "f5": 0.084}`

## Thresholds
n/a

## Status
passed: Внешний payload успешно приведён к AdaptedInput.

## Interpretation
Фреймворк получил уверенность модели, признаки, важности признаков, метрики качества и контекст ограничения.

## Next
explanation_object
