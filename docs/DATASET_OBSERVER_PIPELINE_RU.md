# Dataset Observer Pipeline

Цель слоя: запускать наблюдатель не только на синтетическом примере, а на табличном датасете из локального файла, прямой ссылки или открытого реестра.

Базовая цепочка:

```text
dataset -> data profile -> A_M^F -> model -> E_M^ext -> I_pre -> rho(x) -> E_R -> E_A -> action/report
```

## Источники

`registry.cit.gov.ru/datasets` рендерится динамически, поэтому автоматический парсинг карточек не используется как обязательная часть v1. Поддерживаются два устойчивых режима:

- локальный файл, скачанный из реестра вручную;
- прямая ссылка на CSV/XLSX/JSON/Parquet, если она есть в карточке набора данных.

## Что вычисляется

1. `DatasetRecord` сохраняет происхождение набора данных.
2. `DatasetProfile` определяет строки, столбцы, числовые/категориальные признаки, пропуски, интервальные пары, экспертные и source-колонки.
3. По профилю выводятся типы неопределённости: `u_num`, `u_ling`, `u_int`, `u_exp`, `u_conf`, `u_trace`, `u_multi`.
4. Селектор выбирает минимально достаточное представление `A_M^F`: `F0`, `FI`, `FH`, `FNsrc` или `FML-audit`.
5. Обучается базовая модель `RandomForest` или `LogisticRegression`.
6. `RiskAwareModel` строит `E_M^ext`, `E_R`, `E_A`, считает `I_pre`, `rho(x)` и безопасное действие.
7. Сохраняется отчёт в `reports/dataset_observer/`.

## Запуск

```bash
PYTHONPATH=. python examples/dataset_observer_demo.py --sample breast_cancer
```

По умолчанию используется режим `--mode user`: если в датасете нет интервальных, экспертных или конфликтных метаданных, выбирается `A_M^F = F0`.

Локальный файл:

```bash
PYTHONPATH=. python examples/dataset_observer_demo.py --file data/my_dataset.csv --target target
```

Прямая ссылка:

```bash
PYTHONPATH=. python examples/dataset_observer_demo.py --url "https://example.org/dataset.csv" --target target
```

Проверка на Breast Cancer Wisconsin CSV:

```bash
PYTHONPATH=. python examples/dataset_observer_demo.py \
  --url https://raw.githubusercontent.com/Sheikh-talha01/Datasets/main/breast_cancer_data.csv
```

Ожидаемый смысл отчёта: `diagnosis` определяется как целевая колонка, а выбранное представление остаётся `F0`, потому что специальных метаданных неопределённости нет.

Для демонстрации более сложных представлений без большого CIT-архива можно искусственно добавить метаданные:

```bash
PYTHONPATH=. python examples/dataset_observer_demo.py \
  --url https://raw.githubusercontent.com/Sheikh-talha01/Datasets/main/breast_cancer_data.csv \
  --simulate-intervals --simulate-experts --simulate-conflict --mode audit
```

Такой запуск добавляет `*_min/*_max`, `expert_*` и `source_*` поля, поэтому селектор главы 3 выбирает многоуровневое представление.

## Соответствие диссертации

- Глава 2: композиционный контур и диагностические состояния.
- Глава 3: профиль ситуации и выбор формы неопределённости.
- Наблюдатель: риск автоматического применения и безопасное действие.
