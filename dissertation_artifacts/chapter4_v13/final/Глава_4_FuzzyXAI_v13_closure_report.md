# Closure report главы 4 v13

## Проверяемые пункты

- бюджеты 5/10/20/30/40 %: `PASS`; строк: `5`; знак эффекта: `baseline_error_rate - fuzzyxai_error_rate`;
- runtime N=1/10/100/1000 и N=10000 для маскирования: `PASS`; конфигураций: `9`; сырых повторов: `45`; прогревов: `1`;
- runtime median/mean/std/p95/p99, RAM и VRAM: `PASS`; повторов на конфигурацию: `5`;
- held-out faults: `EXPLORATORY`; объектов: `1000` на метод; типы заранее зафиксированы, но зарегистрированы в контракте валидатора; это не open-set проверка произвольных отказов;
- leakage audit: `PASS`;
- числовых записей evidence map: `868`;
- код: MIT (`LICENSE` и `pyproject.toml`); лицензии данных и модели: `THIRD_PARTY_NOTICES.md`;
- полное воспроизведение: `make reproduce-chapter4-v13 CHAPTER4_V13_PYTHON=/path/to/python3.12`;
- лёгкая CI-проверка: `make chapter4-v13-smoke CHAPTER4_V13_PYTHON=/path/to/python3.12`.

## Граница held-out проверки

exploratory; no universal unknown-fault detection claim. Пять типов были исключены из настройки простых baseline, но их поля и проверки присутствуют в типизированной схеме. Универсальное обнаружение неизвестного класса отказа не заявляется.

## Контрольные суммы

- `8d8e2f7c121356ada87fad384b36c40d26779af054aca0db21b4d7519bb22a80`  `Глава_4_FuzzyXAI_v13_budget_closure.csv`
- `d16324d5ee6b77150e2243ff7fd9d620268374cf32ee88c73c10d8866e67e2b5`  `Глава_4_FuzzyXAI_v13_runtime_summary_full.csv`
- `c29c1b06ef929819bb9b87fe1b60f4bd6241bfc9b1a83ccb3c395a38024b7183`  `Глава_4_FuzzyXAI_v13_runtime_raw_results.csv`
- `236a2f71136f22c099f1244f4e7582ec4b51e5cad0c56bf59ce5df90453dfd6c`  `Глава_4_FuzzyXAI_v13_held_out_faults.csv`

## Публичная линия Git

- remote: `https://github.com/lebedeffson/fuzzy-xai-operator.git`;
- experiment branch: `experiments/chapter4-practical-v13`;
- stable tag target: `v1.3.0 -> 1a71bae98f1554430d537670018dce7dc889e25f`;
- `v1.3.0` не перемещается при публикации closure-артефактов.
