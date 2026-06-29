# Repo Restructure Plan: DubnaXAI

## 1. Цель

Разделить текущий репозиторий на три самостоятельных слоя:

```text
framework/      библиотека FuzzyXAI для установки и повторного использования
site/           сайт DubnaXAI как исследовательская витрина
applications/   прикладные сценарии и воспроизводимые задачи
```

Итоговая логика для диссертации:

```text
глава 4 -> framework/fuzzyxai
глава 5 -> site/dubnaxai
глава 6 -> applications/scenarios
```

На этом шаге код не переносится. Файл фиксирует инвентаризацию и безопасную карту миграции.

## 2. Текущее состояние

| Зона | Файлов | Размер | Статус |
|---|---:|---:|---|
| `fuzzyxai/` | 338 | 1.36 MB | ядро, audit, train/evaluate, practice, studio смешаны |
| `apps/` | 17 | 0.68 MB | интерфейсы и демо-приложения |
| `configs/` | 9 | 0.07 MB | сценарии и ExplainPlan |
| `reports/practice_demo/` | 94 | 8.46 MB | практический пакет, скрины, proof, inputs, QA |
| `reports/real_validation/` | 3 | small | проверка real public artifacts |
| `data/real_public/` | 4 | 2.56 MB | iris + PhysioNet ECG |
| `docs/` | 21 | 4.03 MB | главы, методические документы |
| `tests/audit/` | 36 | 0.23 MB | readiness/audit tests |

Основные релизные архивы:

| Архив | Файлов | Размер | Назначение |
|---|---:|---:|---|
| `FuzzyXAI_full_delivery_package.zip` | 136 | 29.93 MB | полный delivery package |
| `fuzzyxai_doctoral_runtime_release.zip` | 250 | 10.69 MB | runtime-код и audit package |
| `fuzzyxai_final_audit_package.zip` | 97 | 5.85 MB | audit-артефакты |
| `visual_artifacts_latest.zip` | 43 | 2.42 MB | визуальные материалы |

## 3. Целевая структура

```text
DubnaXAI/
├── framework/
│   └── fuzzyxai/
│       ├── pyproject.toml
│       ├── README.md
│       ├── LICENSE
│       ├── fuzzyxai/
│       ├── examples/
│       └── tests/
├── site/
│   └── dubnaxai/
├── applications/
│   └── scenarios/
├── data/
├── docs/
├── reports/
└── README.md
```

## 4. Карта переноса

| Сейчас | Куда | Глава | Действие |
|---|---|---:|---|
| `fuzzyxai/core/` | `framework/fuzzyxai/fuzzyxai/core/` | 4 | перенести как вычислительное ядро |
| `fuzzyxai/adapters/` | `framework/fuzzyxai/fuzzyxai/adapters/` | 4 | перенести |
| `fuzzyxai/hierarchy/` | `framework/fuzzyxai/fuzzyxai/uncertainty/` или оставить `hierarchy/` | 4 | решить при миграции API |
| `fuzzyxai/risk/` | `framework/fuzzyxai/fuzzyxai/risk/` | 4 | перенести |
| `fuzzyxai/selection/` | `framework/fuzzyxai/fuzzyxai/selection/` | 4 | перенести |
| `fuzzyxai/rules/` | `framework/fuzzyxai/fuzzyxai/rules/` | 4 | перенести |
| `fuzzyxai/trust/` | `framework/fuzzyxai/fuzzyxai/trust/` | 4 | перенести |
| `fuzzyxai/core/proof_package.py` | `framework/fuzzyxai/fuzzyxai/proof/` | 4 | выделить доказательный след |
| `fuzzyxai/audit/` | `reports/tools/` + частично `framework/fuzzyxai/fuzzyxai/audit/` | 4/6 | разделить runtime verifier и диссертационный audit |
| `fuzzyxai/train/` | `applications/scenarios/*/train.py` или `applications/tools/train/` | 6 | вынести из библиотеки |
| `fuzzyxai/evaluate/` | `applications/scenarios/*/evaluate.py` или `applications/tools/evaluate/` | 6 | вынести из библиотеки |
| `fuzzyxai/realdata/` | `applications/real_public/tools/` | 6 | оставить как прикладной слой |
| `fuzzyxai/studio/` | `site/dubnaxai` + `applications/tools/studio/` | 5/6 | разделить сайт и локальный стенд |
| `apps/fuzzyxai_studio.py` | `applications/tools/studio/fuzzyxai_studio.py` | 6 | перенести как стенд сценариев |
| `configs/studio_scenarios/` | `applications/scenarios/*/config/` | 6 | разложить по сценариям |
| `reports/practice_demo/scenario_inputs/` | `applications/scenarios/*/input/` | 6 | разложить по scenario_id |
| `reports/practice_demo/model_cards/` | `applications/scenarios/*/model_card/` + `site/dubnaxai/src/content/models/` | 5/6 | копия для сайта, источник в applications |
| `reports/practice_demo/proof_packages/` | `applications/scenarios/*/proof/` | 6 | разложить по сценариям |
| `reports/practice_demo/tables/` | `applications/scenarios/*/tables/` | 6 | разложить |
| `reports/practice_demo/screenshots/` | `applications/scenarios/*/screenshots/` + `docs/assets/screenshots/` + `site/dubnaxai/public/screenshots/` | 5/6 | копировать в нужные потребители |
| `reports/practice_demo/chapter_tables/` | `docs/assets/tables/` | 6 | таблицы для глав |
| `reports/practice_demo/qa/` | `reports/validation/practice_demo/` | 6 | отчёты проверки |
| `reports/real_validation/` | `reports/validation/real_public/` | 6 | отчёты real public validation |
| `data/real_public/` | `data/real_public/` | 6 | оставить как данные |
| `docs/chapters/` | `docs/chapters/` | 4-6 | оставить |
| `figures/`, `visual_artifacts_latest.zip` | `docs/assets/figures/` | 4-6 | распаковать/перенести отобранное |
| `tests/audit/` | `reports/tools/tests/` + `framework/fuzzyxai/tests/` | 4/6 | разделить framework tests и delivery gates |
| `.github/workflows/` | `.github/workflows/` | release | оставить в корне |

## 5. Сценарии applications

Каждый сценарий должен стать отдельным воспроизводимым блоком:

```text
applications/scenarios/<scenario_id>/
├── README.md
├── input/
├── config/
├── model_card/
├── run.py
├── proof/
├── tables/
└── screenshots/
```

Первая партия:

| Scenario | Источник сейчас | Evidence level | Цель |
|---|---|---|---|
| `hybrid_xiris` | `reports/practice_demo/*hybrid*`, `configs/studio_scenarios/hybrid_xiris.json` | `full_control_run` | радужка, batch, block |
| `medical_ecg_signal` | `reports/practice_demo/*ecg*`, `data/real_public/ecg/` | `operator_control_example` | ECG, quality, expert |
| `gd_anfis_shap` | `reports/practice_demo/*gd*` | `operator_control_example` | правило vs SHAP |
| `beacon_xai` | `reports/practice_demo/*beacon*` | `route_demonstration` | time counterevidence |
| `gis_integro` | `reports/practice_demo/*gis*` | `operator_control_example` | geolayer, route alignment |

## 6. Framework API, который надо привести

Сейчас `fuzzyxai/__init__.py` смешивает старые и новые сущности, а также содержит дубль импорта `reduce_to_f0`.

Целевой публичный слой:

```python
from fuzzyxai import (
    build_explanation,
    compute_alignment,
    compute_reduction_loss,
    observe_risk,
    diagnose,
    make_action,
    build_proof_trace,
    verify_proof_trace,
    show_operator_route,
)
```

Задачи для `framework/fuzzyxai`:

1. Сохранить обратную совместимость для старых импортов.
2. Добавить чистый top-level API.
3. Отделить `proof`, `risk`, `diagnostics`, `actions`.
4. Вынести train/evaluate/practice/studio из библиотеки.
5. Проверка:

```bash
pip install -e framework/fuzzyxai
python -c "import fuzzyxai; print(fuzzyxai.__version__)"
pytest framework/fuzzyxai/tests -q
```

## 7. Site DubnaXAI

Создать `site/dubnaxai` как отдельный статический проект.

Минимальные страницы:

```text
/
/researchers
/models
/methods
/publications
/demos
/operators
/about
```

Данные сайта должны читаться из структурированных файлов:

```text
site/dubnaxai/src/data/models.json
site/dubnaxai/src/data/methods.json
site/dubnaxai/src/data/researchers.json
site/dubnaxai/src/data/publications.json
```

Начальные источники данных:

| Site data | Источник |
|---|---|
| `models.json` | `reports/practice_demo/model_cards/` |
| `methods.json` | `fuzzyxai/core`, `fuzzyxai/risk`, `fuzzyxai/hierarchy`, README |
| `demos.json` | `reports/practice_demo/practice_manifest.json` |
| screenshots | `reports/practice_demo/screenshots/` |

## 8. Что не переносить в framework

В `framework/fuzzyxai` не должны попасть:

```text
reports/
docs/chapters/
*.zip
screenshots/
practice_demo/
data/real_public/
models/
dissertation_artifacts/
patches/
```

Исключение: маленькие examples/tests fixtures, если они нужны для unit tests.

## 9. Generated / dirty artifacts

Сейчас рабочее дерево содержит сгенерированные артефакты:

```text
FuzzyXAI_FINAL_DELIVERY_REPORT.md
FuzzyXAI_full_delivery_package.zip
data/real_public/
models/
reports/dataset_audit/
reports/evaluation/
reports/practice_demo/
reports/real_validation/
reports/training/
reports/audit/*
reports/studio_batch/*
visual_artifacts_latest.zip
```

Их нельзя случайно смешивать с исходным framework-кодом. При миграции:

1. Сначала создать структуру.
2. Затем копировать только нужные артефакты в `applications/`, `docs/assets/`, `reports/validation/`.
3. После копирования запускать проверки.
4. Только затем удалять или архивировать старые места.

## 10. Порядок работ

### Этап 0. Зафиксировать план

```bash
git add REPO_RESTRUCTURE_PLAN.md
git commit -m "Add DubnaXAI restructure plan"
```

### Этап 1. Создать пустую структуру

```text
framework/fuzzyxai/
site/dubnaxai/
applications/scenarios/
data/
docs/
reports/
```

Без удаления старых файлов.

### Этап 2. Выделить framework

1. Скопировать `fuzzyxai/core`, `adapters`, `hierarchy`, `risk`, `rules`, `selection`, `trust`.
2. Создать `framework/fuzzyxai/pyproject.toml`.
3. Создать чистый `framework/fuzzyxai/fuzzyxai/__init__.py`.
4. Перенести только framework tests.
5. Проверить `pip install -e framework/fuzzyxai`.

### Этап 3. Разложить applications

1. Создать пять папок сценариев.
2. Разложить inputs, model cards, proof, tables, screenshots.
3. Для каждого сценария создать `README.md` и `run.py`.
4. Проверить один общий command:

```bash
python applications/scenarios/hybrid_xiris/run.py
python applications/scenarios/medical_ecg_signal/run.py
```

### Этап 4. Создать site

1. Статический сайт `site/dubnaxai`.
2. Данные из JSON.
3. Скриншоты из `site/dubnaxai/public/screenshots`.
4. Проверка сборки.

### Этап 5. Delivery gate

Создать новую команду:

```bash
make dubnaxai-release-check
```

Она должна проверять:

```text
framework install PASS
framework tests PASS
applications scenario runs PASS
site build PASS
proof verification PASS
reports validation PASS
```

## 11. Риски миграции

| Риск | Как закрываем |
|---|---|
| Потерять working practice package | сначала копируем, потом проверяем, потом чистим |
| Сломать импорты `fuzzyxai` | сохраняем старый пакет до прохождения framework tests |
| Смешать generated reports с source | generated кладём только в `reports/` и `applications/*/proof` |
| Сделать сайт статичной картинкой | сайт читает `src/data/*.json` |
| Сделать главу 6 без воспроизводимости | каждый сценарий получает `run.py` |

## 12. Definition of Done для переразборки

```text
framework/fuzzyxai устанавливается через pip install -e
site/dubnaxai собирается
applications/scenarios/* имеют одинаковую структуру
каждый scenario имеет README, input, model_card, proof, tables, screenshots, run.py
real_public данные лежат отдельно от demo/control данных
reports/validation содержит QA/audit outputs
старый монолитный practice_demo больше не является единственным источником истины
главы 4-6 имеют прямую карту на framework/site/applications
```

