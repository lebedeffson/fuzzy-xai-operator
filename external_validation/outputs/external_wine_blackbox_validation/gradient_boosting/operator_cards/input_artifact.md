# Внешняя табличная модель

## Input
- `source_type` = `"tabular"`
- `model_name` = `"GradientBoostingClassifier"`
- `dataset_name` = `"sklearn_wine"`
- `model_output` = `{"predicted_class": 0, "class_probability": 0.679119}`
- `quality_metrics` = `{"missing_rate": 0.0, "feature_range_violation": 0.0}`
- `feature_values` = `{"alcohol": 13.07, "malic_acid": 1.5, "ash": 2.1, "alcalinity_of_ash": 15.5, "magnesium": 98.0, "total_phenols": 2.4, "flavanoids": 2.64, "nonflavanoid_phenols": 0.28, "proanthocyanins": 1.37, "color_intensity": 3.7, "hue": 1.18, "od280/od315_of_diluted_wines": 2.69, "proline": 1020.0}`
- `attribution_values` = `{"proline": 0.326532, "color_intensity": 0.243624}`
- `context_values` = `{"external_task": true, "model_key": "gradient_boosting", "top_k_importance": 2, "sample_index": 14, "train_size": 115, "test_size": 63}`

## Formula
Нормализация внешнего входа

## Components
- `model_name` = `"GradientBoostingClassifier"`
- `dataset_name` = `"sklearn_wine"`
- `source_type` = `"tabular"`

## Output
- `adapted_input_id` = `"external_wine_classification:GradientBoostingClassifier"`
- `trace_origin` = `"external_model_payload"`
- `class_probability` = `0.679119`
- `feature_importance` = `{"proline": 0.326532, "color_intensity": 0.243624}`

## Thresholds
n/a

## Status
passed: Внешний payload успешно приведён к AdaptedInput.

## Interpretation
Фреймворк получил вероятность класса, признаки, важности признаков и метрики качества.

## Next
explanation_object
