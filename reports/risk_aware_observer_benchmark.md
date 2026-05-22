# Risk-Aware Observer benchmark

Датасет: `sklearn breast_cancer` (569 объектов, 30 признаков).
Модель: `RandomForestClassifier`.

## Базовая модель

- Accuracy: `0.965035`
- ROC-AUC: `0.99434`

## Наблюдатель

- Accepted accuracy: `0.972477`
- Coverage: `0.762238`
- Defer rate: `0.0`
- Request rate: `0.118881`
- Expected cost before: `0.174825`
- Expected cost after: `0.108462`
- Risk reduction: `0.066364`
- Mean I(E): `0.798902`
- Forced conflict action: `block`

## Распределение действий

- `accept`: 109
- `lower_confidence`: 17
- `request_more_data`: 17

## Итог

Наблюдатель не меняет RandomForest, а управляет допустимостью автоматического применения прогноза через риск, неопределённость, интерпретируемость и D_ij.
