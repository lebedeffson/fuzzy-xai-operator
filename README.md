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

## Главная демо

Запуск GUI для показа на защите:

```bash
pip install -r requirements.txt
python apps/defense_demo.py --port 8085
```

Открыть в браузере:

```text
http://localhost:8085
```

Сценарий демо:

1. Обучается реальная модель `sklearn LogisticRegression` по признакам `age`, `pressure`, `marker`.
2. Выбранный кейс проходит через модель и получает `risk_score = predict_proba(...)`.
3. Риск модели переводится в лингвистические термы: `low`, `medium`, `high`.
4. Для кейса строится объяснение `E_k` и выбирается класс представления `A_k^F`.
5. Система проверяет, согласованы ли модель риска и модуль принятия решения.
6. Отдельный benchmark-блок сравнивает режимы на `sklearn breast_cancer` с `RandomForestClassifier`: только модель против модели с нечётким системным оператором.

Переключатель конфликта специально ломает интерфейс между компонентами. В этом режиме система показывает `D_ij`, а не скрывает рассогласование красивым, но неверным отчётом.

GUI рассчитан на показ неспециалисту. Он показывает маршрут `model -> risk_score -> ExplainPlan -> E_k -> A_k^F -> D_ij / I(E_G)`, текущий кейс, вклад признаков модели, функции принадлежности, распределение классов, выбранное представление и проверку согласованности между моделью и решающим модулем.

Для показа на проекторе есть переключатель презентационного режима. Кнопка справки открывает короткую экскурсию. Кнопка печати позволяет сохранить страницу как PDF через браузер.

Ключевые визуальные блоки:

- Метка текущего кейса на функциях принадлежности: видно, куда попал риск пациента относительно `low`, `medium`, `high`.
- Граф вкладов модели: видно, какие признаки подняли или снизили предсказанный риск.
- Граф `A_k^F` по слоям: отдельно показаны интервальная неопределённость, экспертные оценки и конфликт `T/I/F`.
- Граф выбора класса из главы 3: кандидаты расположены по когнитивной сложности и ожидаемой потере редукции, выбранный класс подсвечен.
- Граф композиции: модель и модуль решения соединены стрелкой рассогласования; конфликтный режим приводит к `D_ij`.
- Benchmark “без оператора / с оператором”: baseline-модель даёт риск и важности признаков, а оператор добавляет `gamma`, `I(E_G)` и обнаружение конфликта `D_ij`.

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

В GUI `apps/defense_demo.py` этот отчёт показан прямо на главной странице первым блоком. Оттуда же можно пересобрать отчёт и скачать HTML/JSON.

## Технический dashboard

Расширенный NiceGUI dashboard остаётся доступен для отладки и экспериментов:

```bash
python apps/nicegui_dashboard.py --port 8080
```

Его стоит использовать для загрузки CSV, синтеза `FML`, просмотра отчётов, экспорта сессий и проверки примеров диссертации. Для презентации лучше использовать `apps/defense_demo.py`.


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
```

Отчёты:

- `reports/dataset_observer/dataset_observer_report.json`
- `reports/dataset_observer/dataset_observer_report.md`
- `reports/dataset_observer/dataset_observer_report.html`

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
```

Финальная проверка диссертационных примеров:

```bash
PYTHONPATH=. python proofs/validate_thesis_examples.py
PYTHONPATH=. python examples/thesis_demo.py
```

Ожидаемый статус:

```text
58 passed
thesis validation: PASS
thesis demo: PASS
```

## Отчёты

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
  hierarchy/     F0, interval, hesitant, neutrosophic, multilevel классы
  selection/     построение профиля, совместимость, Парето-выбор
  calibration/   калибровка beta и кросс-валидация
  data/          загрузка датасетов, CIT/local adapters, профиль данных
  pipelines/     dataset observer pipeline
  risk/          Risk-Aware Observer: неопределённость, rho(x), политика, метрики, observer pipeline
  rules/         LOFO-F1 и стабильный отбор правил
  visual/        Plotly-графики функций принадлежности и композиции
  demo/          детерминированные demo-данные и сборщики примеров
apps/
  defense_demo.py       главный GUI для презентации
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

## Научные ограничения

Репозиторий является воспроизводимым исследовательским прототипом для диссертации, а не сертифицированной клинической системой. Медицинский benchmark используется как техническая проверка применимости. Полная доменная валидация, экспертная оценка и регулируемое внедрение находятся вне рамок этого прототипа.
