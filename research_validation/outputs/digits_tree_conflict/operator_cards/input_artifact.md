# Внешняя модель

## Input
- `source_type` = `"tabular"`
- `model_name` = `"DecisionTreeClassifier"`
- `dataset_name` = `"sklearn_digits"`
- `model_output` = `{"predicted_class": 1, "class_probability": 0.83}`
- `quality_metrics` = `{"missing_rate": 0.0, "feature_range_violation": 0.0}`
- `feature_values` = `{"f1": 0.1, "f2": 0.2, "f3": 0.3, "f4": 0.4, "f5": 0.5}`
- `attribution_values` = `{"f1": 0.2272, "f2": 0.1704, "f3": 0.1278, "f4": 0.0994, "f5": 0.0852}`
- `context_values` = `{"experiment_id": "digits_tree_conflict", "task_type": "tabular_classification", "perturbation": "explanation_conflict", "representation_class": "NAS", "prediction_interval_width": 0.0, "conflict_component": 0.67, "noise_ratio": 0.0, "occlusion_rate": 0.0}`

## Formula
Нормализация внешнего входа

## Components
- `model_name` = `"DecisionTreeClassifier"`
- `dataset_name` = `"sklearn_digits"`
- `source_type` = `"tabular"`
- `task_type` = `"tabular_classification"`
- `perturbation` = `"explanation_conflict"`

## Output
- `adapted_input_id` = `"external_wine_classification:DecisionTreeClassifier"`
- `trace_origin` = `"external_model_payload"`
- `class_probability` = `0.83`
- `feature_importance` = `{"f1": 0.2272, "f2": 0.1704, "f3": 0.1278, "f4": 0.0994, "f5": 0.0852}`

## Thresholds
n/a

## Status
passed: Внешний payload успешно приведён к AdaptedInput.

## Interpretation
Фреймворк получил уверенность модели, признаки, важности признаков, метрики качества и контекст ограничения.

## Next
explanation_object
