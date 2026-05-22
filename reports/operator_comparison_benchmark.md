# Benchmark: без оператора / с оператором

Датасет: `sklearn breast_cancer` (569 объектов, 30 признаков).
Модель: `RandomForestClassifier`.

## Качество модели

- Accuracy: `0.965035`
- ROC-AUC: `0.99434`

## Что есть без оператора

- Доступно: risk_score, accuracy, roc_auc, feature_importance.
- Отсутствует: semantic_gamma, interpretability_index, D_ij_conflict_detection.
- Вывод: Модель объясняет риск локально, но не проверяет согласованность цепочки model -> decision.

## Что добавляет оператор

- Проверено кейсов: `50`
- Среднее `gamma`: `0.172724`
- Средний `I(E_G)`: `0.764437`
- Доля обнаружения искусственного конфликта `D_ij`: `1.0`
- Вывод: Оператор добавляет проверку семантической совместимости и диагностирует разрыв терминов.

## Топ признаков модели

- `worst area`: 0.135892
- `worst perimeter`: 0.135061
- `worst concave points`: 0.105901
- `mean concave points`: 0.101148
- `worst radius`: 0.09025
- `mean radius`: 0.06832
- `mean perimeter`: 0.068292
- `mean concavity`: 0.040909
- `mean area`: 0.040515
- `worst concavity`: 0.028075

## Итог

FuzzyXAI не повышает accuracy модели напрямую. Его вклад — проверка объяснительной цепочки: gamma, I(E_G), D_ij и воспроизводимый trace.
