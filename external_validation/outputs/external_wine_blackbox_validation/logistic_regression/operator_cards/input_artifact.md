# Внешняя табличная модель

## Input
- `source_type` = `"tabular"`
- `model_name` = `"LogisticRegression"`
- `dataset_name` = `"sklearn_wine"`
- `model_output` = `{"predicted_class": 1, "class_probability": 0.689724}`
- `quality_metrics` = `{"missing_rate": 0.0, "feature_range_violation": 0.0}`
- `feature_values` = `{"alcohol": 12.29, "malic_acid": 1.61, "ash": 2.21, "alcalinity_of_ash": 20.4, "magnesium": 103.0, "total_phenols": 1.1, "flavanoids": 1.02, "nonflavanoid_phenols": 0.37, "proanthocyanins": 1.46, "color_intensity": 3.05, "hue": 0.906, "od280/od315_of_diluted_wines": 1.82, "proline": 870.0}`
- `attribution_values` = `{"alcohol": 0.166384, "proline": 0.145116, "color_intensity": 0.133733, "ash": 0.0985, "hue": 0.083139}`
- `context_values` = `{"external_task": true, "model_key": "logistic_regression", "top_k_importance": 5, "sample_index": 24, "train_size": 115, "test_size": 63}`

## Formula
Нормализация внешнего входа

## Components
- `model_name` = `"LogisticRegression"`
- `dataset_name` = `"sklearn_wine"`
- `source_type` = `"tabular"`

## Output
- `adapted_input_id` = `"external_wine_classification:LogisticRegression"`
- `trace_origin` = `"external_model_payload"`
- `class_probability` = `0.689724`
- `feature_importance` = `{"alcohol": 0.166384, "proline": 0.145116, "color_intensity": 0.133733, "ash": 0.0985, "hue": 0.083139}`

## Thresholds
n/a

## Status
passed: Внешний payload успешно приведён к AdaptedInput.

## Interpretation
Фреймворк получил вероятность класса, признаки, важности признаков и метрики качества.

## Next
explanation_object
