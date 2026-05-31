# FuzzyXAI Doctoral Core

Исполняемый исследовательский прототип для глав 2 и 3 докторской диссертации.

Проект показывает системный оператор нечёткой логики для композиционной объяснимости и иерархию нечётких представлений неопределённости. Это не просто набор формул: математические объекты можно построить, редуцировать, сравнить, скомпоновать, откалибровать, визуализировать и проверить воспроизводимыми скриптами.

## Что реализовано

Глава 2: системный оператор объяснимости

- `ExplanationObject`: объект объяснения `E_k`.
- `ExplainPlan`: воспроизводимая конфигурация термов, весов, порогов и метаданных.
- `SystemOperator`: реализация оператора для скалярного риска.
- `compose`: композиция объектов объяснения с диагностическим состоянием `D_ij`.
- `semantic_disagreement`: семантическое рассогласование `d_E` и расширенное рассогласование с потерей редукции.
- `interpretability_loss` и `interpretability_index`: `L(E)` и `I(E_G)`.
- Калибровка весов `beta` и отчёты кросс-валидации.
- Визуальная диагностика композиции объяснений.

Глава 3: иерархия нечётких представлений

- `F0`: классическое нечёткое представление.
- `IntervalFS`: интервальное нечёткое представление.
- `HesitantFS`: несколько экспертных оценок.
- `NeutrosophicFS`: истинность, неопределённость и ложность с учётом источников.
- `MultiLevelFS`: многоуровневое представление.
- Редукции к `F0` и потеря редукции `Delta`.
- Ситуационный профиль `P_sit`.
- Парето-выбор минимально достаточного класса представления.
- Диагностика выбора `D_choice`.

## Актуальный статус (2026-05-29)

Ниже текущее состояние именно практической части для глав 2-3.

Что уже есть:

- Полный риск-наблюдатель с действиями `accept/lower_confidence/request_more_data/defer_to_human/block`.
- Калибровка наблюдателя и ablation-бенчмарк.
- Baseline comparison с двумя режимами доступа:
  - `baseline_access=native`
  - `baseline_access=equal_guardrail`
- Structure-aware benchmark для контролируемых perturbation-сценариев.
- Defense-cases и автоматический экспорт thesis-таблиц в LaTeX.

Текущие dataset modes:

- built-in:
  - `breast_cancer`
  - `diabetes_binary`
  - `wine_risk`
  - `synthetic_ruptures`
- registry:
  - `registry_programs`
  - `registry_mosmed_doctor_analysis`
  - `registry_steel_ir`

Роли датасетов:

- `breast_cancer`: количественная медицинская демонстрация контура.
- `wine_risk`: переносимость на другой табличный домен.
- `synthetic_ruptures`: safety/diagnostic стенд для разрывов.
- `diabetes_binary`: stress-test калибровки.
- `registry_*`: external-transfer/readiness, без завышенных claims по качеству.

Ключевой safety-результат:

- Для `synthetic_ruptures`:
  - `missed_critical_ruptures = 0`
  - `critical_rupture_recall = 1.0`
  - `false_auto_accept_rate = 0.0`
- См.:
  - `reports/datasets/synthetic_ruptures/summary.json`

Почему два baseline-режима:

- `equal_guardrail`: всем методам передаётся внешний `chi_R_crit` (sanity-check).
- `native`: baseline получают только свой естественный вход:
  - threshold: `proba`
  - SHAP/LIME/Anchors: `proba` + их локальные объяснения
  - без `chi_R`, `chi_R_crit`, `chi_Auto`, `Delta`, trace, CertifiedPath
- Full observer использует полный структурный вход.

Главный сравнительный артефакт:

- `reports/thesis_tables/table_synthetic_guardrail_modes.tex`
  - в `native` baseline пропускают критические разрывы,
  - в `equal_guardrail` блокируют их при наличии внешнего safety-сигнала.

Быстрый запуск:

```bash
make dataset-modes-check
make benchmark-dataset DATASET=synthetic_ruptures
make baseline-comparison DATASET=synthetic_ruptures BASELINE_ACCESS=native
make baseline-comparison DATASET=synthetic_ruptures BASELINE_ACCESS=equal_guardrail
make calibrate-observer DATASET=breast_cancer
make ablation-benchmark DATASET=breast_cancer
make structure-aware-benchmark DATASET=breast_cancer
make thesis-practice-tables
make dissertation-check
make ui-health-check
```

Оперативный статус dataset modes (фактически из `make dataset-modes-check`):

| dataset_mode | status | rows | domain | validates |
| --- | --- | ---: | --- | --- |
| breast_cancer | READY | 569 | medical | risk observer baseline |
| diabetes_binary | READY | 442 | medical | borderline uncertainty |
| wine_risk | READY | 178 | tabular | transferability |
| synthetic_ruptures | READY | 900 | diagnostic | controlled ruptures |
| registry_programs | READY | 10007 | text-tabular | registry records |
| registry_mosmed_doctor_analysis | READY | 10 | medical audit | doctor/model audit |
| registry_steel_ir | READY | 8080 | industrial CV | industrial transferability |

Ключевые практические артефакты (готовы):

- Dataset summaries/cards: `reports/datasets/<dataset>/{summary.json,summary.md,data_card.md,predictions.csv}`
- Calibration (Breast Cancer): `reports/datasets/breast_cancer/calibration.json`
- Ablation:
  - `reports/datasets/breast_cancer/ablation.json`
  - `reports/datasets/synthetic_ruptures/ablation.json`
- Baseline comparison:
  - `reports/datasets/*/baseline_comparison.json`
  - `reports/datasets/synthetic_ruptures/baseline_comparison_native.json`
  - `reports/datasets/synthetic_ruptures/baseline_comparison_equal_guardrail.json`
- Defense cases: `reports/defense_cases/{accept_case.json,audit_case.json,block_case.json,summary.md}`
- Structure-aware benchmark: `reports/structure_aware_benchmark/breast_cancer.json`
- Thesis practice index: `reports/thesis_practice/thesis_practice_tables.md`
- LaTeX tables: `reports/thesis_tables/*.tex`

## Главная демо

Единый вход для интерактивной демонстрации (Studio):

```bash
make demo PORT=8097
# aliases:
# make defense-demo PORT=8097
# make studio PORT=8097
# direct:
PYTHONPATH=. python apps/fuzzyxai_studio.py --port 8097
```

`FuzzyXAI Studio` объединяет в одной странице:

- `Защита`: сквозной маршрут `Dataset -> E_k -> A_k^F -> chi_Auto -> rho -> Action`.
- `Эксперт`: интерактивный ExplainPlan editor + what-if (веса/пороги/rupture/context/trace).
- `Benchmark`: загрузка `baseline_comparison` (`native/equal_guardrail`) и structure-aware отчётов.
- `Отчёты`: экспорт текущего кейса в JSON/Markdown/LaTeX.
- `Методология`: краткая карта связи с главами 2-3.

Официальная страница для защиты: **только `FuzzyXAI Studio`**.  
`defense_demo.py` и `layered_demo.py` остаются только для internal/debug и помечены как `*-legacy`.

Быстрый pre-defense UI check:

```bash
make ui-health-check
# legacy apps (опционально):
make ui-health-check-all
```

Артефакты проверки:

- `reports/ui_health_check.json`
- `reports/ui_health_check.md`

Визуальная браузерная проверка (headless Chromium, со скриншотами вкладок):

```bash
make browser-visual-check
```

Артефакты:

- `reports/browser_visual_check/browser_visual_check.json`
- `reports/browser_visual_check/browser_visual_check.md`
- `reports/browser_visual_check/01_studio_home.png`
- `reports/browser_visual_check/02_case_controls.png`
- `reports/browser_visual_check/03_plan_editor.png`
- `reports/browser_visual_check/04_what_if.png`
- `reports/browser_visual_check/05_benchmark.png`
- `reports/browser_visual_check/06_export.png`
- `reports/browser_visual_check/07_summary.png`

Запуск GUI для показа на защите:

```bash
pip install -r requirements.txt
python apps/fuzzyxai_studio.py --port 8097
```

Открыть в браузере:

```text
http://localhost:8097
```

Сценарий демо:

1. Выбрать `dataset mode`, `scenario`, `sample index`.
2. Нажать `Run pipeline` и получить итог `Action + reason`.
3. В блоке `ExplainPlan` поменять веса/пороги и нажать `Apply plan`.
4. В блоке `What-if` включить override и проверить, как меняется `rho`, `chi_R/chi_R_crit`, `chi_Auto`, `action`.
5. В блоке `Benchmark` загрузить `native` и `equal_guardrail` отчёты.
6. В `Export` сохранить текущий кейс в JSON/MD/TEX.

Страница единая: слева управление (кейс/план/what-if/benchmark/export), справа сквозной результат и трассировка.

Ключевые визуальные блоки:

- Membership view (`low/medium/high`) с текущей точкой кейса.
- Risk contribution view по компонентам `predicted_risk/uncertainty/interpretability_gap/reduction_loss/chi_R`.
- Pipeline route (`Dataset -> E_k -> A_k^F -> chi_Auto -> rho -> Action`).
- CertifiedPath/Rupture таблица переходов.
- Representation selection таблица кандидатов и выбранного класса.
- Raw trace JSON для аудита.

## Полная демонстрация

Сквозной сценарий для презентации всей логики одной командой:

```bash
make full-demo
# или
PYTHONPATH=. python full_pipeline_demo.py --open
```

Что показывает отчёт:

- загрузка медицинских demo-данных и обучение `LogisticRegression`;
- разбор одного кейса: признаки -> `risk_score` -> термы `low/medium/high`;
- системный оператор главы 2: `E_k`, композиция этапов, `gamma`, `D_ij`, `I(E_G)`;
- глава 3: профиль ситуации `P_sit`, выбор класса `A_k^F`, потеря редукции `Delta`;
- Risk-Aware Observer: итоговое действие `accept/lower_confidence/request_more_data/defer_to_human/block`;
- benchmark на `sklearn breast_cancer`, где видно снижение ожидаемой стоимости риска.

Главный файл отчёта:

```text
reports/full_demo/index.html
```

Также сохраняются:

- `reports/full_demo/full_pipeline_report.json`
- `reports/full_demo/full_pipeline_report.md`
- `reports/full_demo/01_memberships.html`
- `reports/full_demo/02_feature_contributions.html`
- `reports/full_demo/03_representation.html`
- `reports/full_demo/04_composition_graph.html`

В GUI `apps/fuzzyxai_studio.py` этот отчёт доступен из режима работы с отчётами и экспорта кейса.

## Технический dashboard

Расширенный NiceGUI dashboard остаётся доступен для отладки и экспериментов:

```bash
python apps/nicegui_dashboard.py --port 8080
```

Его стоит использовать для загрузки CSV, синтеза `FML`, просмотра отчётов, экспорта сессий и проверки примеров диссертации. Для презентации используется `apps/fuzzyxai_studio.py`.


## Risk-Aware XAI Observer

Новый слой над главами 2-3: системный оператор используется как внешний наблюдающий контур над моделью с прогнозным интерфейсом. Он не меняет параметры модели, а строит `E_M^ext`, выбирает `A_M^F`, считает неопределённость, `Delta`, предварительный индекс `I_pre`, риск автоматического применения `rho(x)`, действие и финальный индекс `I_final`.

Ключевая функция риска вынесена в `fuzzyxai/risk/risk_function.py` и совпадает с математической записью:

```text
rho = w_p*rho_p + w_u*u_M + w_I*(1 - I_pre) + w_Delta*Delta_M + w_D*1[D_pre != empty]
```

Действия наблюдателя:

- `accept`
- `lower_confidence`
- `request_more_data`
- `defer_to_human`
- `block`

Запуск центрального контура:

```bash
make full-observer
# или
PYTHONPATH=. python full_observer_pipeline.py --open
```

Отчёты и математическое описание:

- `docs/RISK_AWARE_XAI_OBSERVER_MATH_RU.md`
- `docs/DATASET_OBSERVER_PIPELINE_RU.md`
- `reports/full_observer_pipeline/full_observer_pipeline.json`
- `reports/full_observer_pipeline/full_observer_pipeline.md`
- `reports/full_observer_pipeline/full_observer_pipeline.html`

Минимальный API:

```python
from fuzzyxai.risk import build_full_observer_pipeline_report, compute_application_risk

report = build_full_observer_pipeline_report()
print(report["with_observer"]["safe_action"])

rho = compute_application_risk(0.72, 0.31, 0.84, 0.09, []).rho
print(rho)
```

## Dataset Observer

Слой для реальных табличных датасетов: локальный CSV/XLSX/JSON/Parquet, прямая ссылка на файл из реестра или sample `breast_cancer`. Он строит профиль данных, обучает модель и прогоняет кейс через наблюдатель.

```bash
make dataset-observer
# или
PYTHONPATH=. python examples/dataset_observer_demo.py --sample breast_cancer
PYTHONPATH=. python examples/dataset_observer_demo.py --file data/my_dataset.csv --target target
PYTHONPATH=. python examples/dataset_observer_demo.py --url https://raw.githubusercontent.com/Sheikh-talha01/Datasets/main/breast_cancer_data.csv
```

Для обычного Breast Cancer CSV без специальных метаданных селектор выбирает `F0`. Чтобы показать интервалы, экспертов и конфликт источников на том же файле, можно включить демонстрационные флаги:

```bash
PYTHONPATH=. python examples/dataset_observer_demo.py --url https://raw.githubusercontent.com/Sheikh-talha01/Datasets/main/breast_cancer_data.csv --simulate-intervals --simulate-experts --simulate-conflict --mode audit
```

Отчёты:

- `reports/dataset_observer/dataset_observer_report.json`
- `reports/dataset_observer/dataset_observer_report.md`
- `reports/dataset_observer/dataset_observer_report.html`

## Dataset modes и debug-утилиты

Для защиты используется только `apps/fuzzyxai_studio.py` (`make demo`).
`apps/layered_demo.py` и `apps/defense_demo.py` оставлены только для debug/обратной совместимости.

`Input -> Model -> Omega -> Expl -> Fuzzy -> Topos -> Observer -> Action`.

Запуск:

```bash
make layered-demo-legacy PORT=8096   # debug only
make defense-demo-legacy PORT=8085   # debug only
```

Проверка режимов датасетов:

```bash
make dataset-modes-check
```

Запуск real-data validation (CITR/RIKORD/RuCCoD, с прозрачным fallback при отсутствии локальных файлов):

```bash
make real-data-validation
```

Бенчмарк одного датасета:

```bash
make benchmark-dataset DATASET=breast_cancer
make benchmark-dataset DATASET=rikord
```

Документация по набору: `docs/datasets.md`.

Режимы:

- built-in: `breast_cancer`, `diabetes_binary`, `wine_risk`, `synthetic_ruptures`
- registry: `registry_programs`, `registry_mosmed_doctor_analysis`, `registry_steel_ir`

Для registry-режимов статус `MISSING` означает, что локальный файл ещё не загружен.
Инструкция по путям: `docs/REGISTRY_DATASETS_SETUP_RU.md`.

## Risk-Aware Observer

Добавлен риск-ориентированный наблюдатель модели. Он не меняет модель, а работает как decision gate поверх неё: получает `predict_proba`, оценивает неопределённость, строит `E_k`, учитывает `I_pre`, `Delta`, диагностические состояния, считает `rho(x)` и выбирает безопасное действие.

Действия наблюдателя:

- `accept`: принять прогноз автоматически.
- `lower_confidence`: сохранить класс, но понизить уверенность.
- `request_more_data`: запросить дополнительные данные.
- `defer_to_human`: передать эксперту.
- `block`: заблокировать автоматическое решение при диагностическом разрыве.

Запуск benchmark:

```bash
make risk-benchmark
# или
PYTHONPATH=. python benchmarks/risk_aware_observer_benchmark.py
```

Отчёты:

- `reports/risk_aware_observer_benchmark.json`
- `reports/risk_aware_observer_benchmark.md`

## LOFO-F1 для отбора правил

Добавлен быстрый метод **Leave-One-Rule-Out F1 importance** для KAFN-подобных аддитивных правил. Он оценивает важность правила напрямую через падение F1 на валидации:

```text
z_without_r = z_full - H_val[:, r] * theta[r]
importance_r = F1_full - F1_without_r
```

Переобучение не требуется: вклад правила просто вычитается из уже посчитанных логитов. Это делает метод дешёвым, но более привязанным к целевой метрике, чем эвристика `|theta_r| * mean(|H[:, r]|)`.

Запуск demo:

```bash
make lofo-f1-demo
# или
PYTHONPATH=. python benchmarks/lofo_f1_rule_pruning_demo.py
```

Отчёты:

- `reports/lofo_f1_rule_pruning.json`
- `reports/lofo_f1_rule_pruning.md`

Минимальный API:

```python
from fuzzyxai.rules import lofo_f1_importance, select_top_rules_by_lofo_f1

importance = lofo_f1_importance(H_val, theta, y_val, bias=bias, rule_names=rule_names)
selected = select_top_rules_by_lofo_f1(importance, budget=25)
```

## Проверка воспроизводимости

Запуск всех тестов:

```bash
PYTHONPATH=. pytest -q
# или
make test
```

Запуск proof-скриптов и регенерация отчётов:

```bash
PYTHONPATH=. python proofs/run_all_proofs.py
# или
make proof
make formal-proof
make category-hott-test
make chapter2-breast-cancer-demo
make chapter5-experiments
make chapter5-demo
make chapter5-latex
make web-demo
make unified-demo
make full-pipeline
make figures
make full-experiments
```

Финальная проверка диссертационных примеров:

```bash
PYTHONPATH=. python proofs/validate_thesis_examples.py
PYTHONPATH=. python examples/thesis_demo.py
```

Ожидаемый статус:

```text
120 passed
thesis validation: PASS
thesis demo: PASS
```

## Отчёты

Глава 5: `reports/chapter5/chapter5_experiments.md`, `chapter5_demo.json`, CSV/LaTeX-таблицы и `sensitivity_w_R.html`.

Глава 2 (real-data): `reports/chapter2/chapter2_breast_cancer_summary.md`, `reports/chapter2/i_pre_distribution.html`.

Сквозной full-pipeline (demo для защиты): `reports/full_pipeline/predictions.csv`, `reports/full_pipeline/summary.md`.


Сгенерированные артефакты сохраняются в `reports/`.

Важные файлы:

- `reports/thesis_validation.md`: численная проверка примеров из глав.
- `reports/thesis_demo_report.md`: сквозной отчёт по маршруту демо.
- `reports/thesis_demo_composition_graph.html`: интерактивный граф композиции.
- `reports/chapter2_calibration_report.json`: отчёт калибровки весов `beta`.
- `reports/breast_cancer_benchmark.md`: краткий benchmark на медицинском датасете.
- `reports/operator_comparison_benchmark.md`: сравнение “без оператора / с оператором”.
- `reports/risk_aware_observer_benchmark.md`: benchmark риск-ориентированного наблюдателя.
- `reports/full_demo/index.html`: полный сценарий “данные -> модель -> глава 2 -> глава 3 -> наблюдатель”.
- `reports/dataset_observer/dataset_observer_report.md`: прогон наблюдателя на табличном датасете.
- `reports/formal_theorems/formal_theorem_checks.md`: проверка формальных теорем.
- `reports/lofo_f1_rule_pruning.md`: demo быстрого LOFO-F1 отбора правил.

## Минимальный пример API

```python
from fuzzyxai import FuzzyXAIPipeline

pipe = FuzzyXAIPipeline.from_data(X_train, y_train, mode="audit")
result = pipe.explain_scalar_risk(
    0.72,
    metadata={
        "has_intervals": True,
        "num_experts": 2,
        "source_conflict": True,
        "requires_audit": True,
    },
)

print(result.report)
```

## Структура проекта

```text
fuzzyxai/
  core/          объекты объяснения, оператор, композиция, расстояния
  category/      Expl, диагностическое расширение, предпучки контекстов
  hott/          Path_Expl, Rupture, temporal drift certificates
  hierarchy/     F0, interval, hesitant, neutrosophic, multilevel классы
  selection/     построение профиля, совместимость, Парето-выбор
  calibration/   калибровка beta и кросс-валидация
  data/          загрузка датасетов, CIT/local adapters, профиль данных
  pipelines/     dataset observer pipeline
  risk/          Risk-Aware Observer: неопределённость, rho(x), политика, метрики, observer pipeline
  rules/         LOFO-F1 и стабильный отбор правил
  studio/        unified studio state/presets/charts/report export
  visual/        Plotly-графики функций принадлежности и композиции
  demo/          детерминированные demo-данные и сборщики примеров
apps/
  fuzzyxai_studio.py    единый интерактивный вход для защиты и эксперта
  defense_demo.py       legacy-демо (оставлено для обратной совместимости)
  nicegui_dashboard.py  расширенный технический dashboard
proofs/
  chapter2_operator_proof.py
  chapter2_calibration_proof.py
  chapter3_hierarchy_proof.py
  validate_thesis_examples.py
examples/
  thesis_demo.py
full_pipeline_demo.py
benchmarks/
  breast_cancer_benchmark.py
  operator_comparison_benchmark.py
  risk_aware_observer_benchmark.py
  lofo_f1_rule_pruning_demo.py
tests/
  pytest-проверки ядра и логики GUI
```

## Связь с главами

Файл `CHAPTER_MAPPING.md` показывает прямое соответствие между объектами диссертации и файлами реализации.

Для подготовки к презентации смотри:

- `PRESENTATION.md`: сценарий защиты и что говорить при показе GUI.
- `IMPLEMENTATION_SUMMARY.md`: техническое описание того, что реализовано и где лежит.
- `ROADMAP_RU.md`: перспективные направления и план доведения прототипа до реальной разработки.
- `docs/CATEGORICAL_HOTT_EXTENSION_RU.md`: приложение с категориально-гомотопической интерпретацией.

Category/HoTT отчёт:

- `reports/category_hott/category_hott_checks.md`
- `docs/FORMAL_THEOREMS_CH2_CH3_RU.md`: шесть формальных теорем для усиления глав 2 и 3.
- `docs/TZ_MATH_RISK_AWARE_OBSERVER_VNEXT_RU.md`: vNext ТЗ для математики, наблюдателя, тестов и proof-отчётов.
- `docs/IMPLEMENTATION_INVENTORY_VNEXT_RU.md`: инвентаризация готового и оставшихся задач.

## Научные ограничения

Репозиторий является воспроизводимым исследовательским прототипом для диссертации, а не сертифицированной клинической системой. Медицинский benchmark используется как техническая проверка применимости. Полная доменная валидация, экспертная оценка и регулируемое внедрение находятся вне рамок этого прототипа.
