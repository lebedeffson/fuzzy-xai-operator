# Dataset Observer Pipeline

Цель слоя: запускать наблюдатель не только на синтетическом примере, а на табличном датасете из локального файла, прямой ссылки или открытого реестра.

Базовая цепочка:

```text
dataset -> data profile -> model -> E_M^ext -> I_pre -> rho(x) -> action -> report
```

## Источники

`registry.cit.gov.ru/datasets` рендерится динамически, поэтому автоматический парсинг карточек не используется как обязательная часть v1. Поддерживаются два устойчивых режима:

- локальный файл, скачанный из реестра вручную;
- прямая ссылка на CSV/XLSX/JSON/Parquet, если она есть в карточке набора данных.

## Что вычисляется

1. `DatasetRecord` сохраняет происхождение набора данных.
2. `DatasetProfile` определяет строки, столбцы, числовые/категориальные признаки, пропуски, интервальные пары, экспертные и source-колонки.
3. По профилю выводятся типы неопределённости: `u_num`, `u_ling`, `u_int`, `u_expert`, `u_conf`, `u_trace`.
4. Обучается базовая модель `RandomForest` или `LogisticRegression`.
5. `RiskAwareModel` строит объяснение прогноза, считает `I_pre`, `rho(x)` и безопасное действие.
6. Сохраняется отчёт в `reports/dataset_observer/`.

## Запуск

```bash
PYTHONPATH=. python examples/dataset_observer_demo.py --sample breast_cancer
```

Локальный файл:

```bash
PYTHONPATH=. python examples/dataset_observer_demo.py --file data/my_dataset.csv --target target
```

Прямая ссылка:

```bash
PYTHONPATH=. python examples/dataset_observer_demo.py --url "https://example.org/dataset.csv" --target target
```

## Соответствие диссертации

- Глава 2: композиционный контур и диагностические состояния.
- Глава 3: профиль ситуации и выбор формы неопределённости.
- Наблюдатель: риск автоматического применения и безопасное действие.
