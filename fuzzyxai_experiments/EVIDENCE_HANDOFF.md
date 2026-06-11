# Evidence-пакет глав 4-5: FuzzyXAI Studio

## Репозиторий и версия

- Репозиторий: https://github.com/lebedeffson/fuzzy-xai-operator
- Ветка: `feature/math-aligned-code`
- Commit hash сборки: см. `git log --oneline -1` в полученном репозитории.
- Пакет: `reports/fuzzyxai_experiments_evidence_pack.zip`

## Как проверить пакет

1. Распаковать архив.
2. Перейти в `fuzzyxai_experiments/`.
3. Выполнить `bash run_chapter4_5.sh`.
4. Выполнить `python compare_reports.py`.
5. Выполнить `sha256sum -c checksums.sha256`.
6. Открыть `reports/gui_screenshots/` для проверки GUI.

Ожидаемый результат: все проверки `PASS`.

```text
PASS: ch4_registry_check
PASS: ch4_evidence_manifest
PASS: hybrid_xiris
PASS: beacon_xai
PASS: gis_integro
PASS: gd_anfis_shap
PASS: generated_tables
PASS: checksums
```

## Docker

```bash
cd fuzzyxai_experiments
docker build -t fuzzyxai/evidence:chapter4-5 .
docker run --rm fuzzyxai/evidence:chapter4-5 bash run_chapter4_5.sh
```

Примечание: в текущей локальной среде Docker daemon недоступен, поэтому проверен shell-запуск `bash run_chapter4_5.sh`.

## Структура

- `README.md`, `EVIDENCE_HANDOFF.md`
- `Dockerfile`, `requirements.txt`, `run_all.sh`, `run_chapter4_5.sh`, `compare_reports.py`
- `registry/modules.json`
- `data/raw`, `data/fixtures`, `data/generated`, `data/article_fixtures`
- `reports/chapter4`, `reports/chapter5`, `reports/gui_screenshots`
- `tables`, `logs`, `scripts`, `src`
- `manifest_sha256.json`, `checksums.sha256`

## Какие числа используются в главах 4-5

HYBRID-XIRIS:

- `total_objects = 1000`
- `critical_cases = 168`
- `baseline_missed = 168`
- `fuzzyxai_missed = 0`
- `score_threshold = 0.70`
- `quality_threshold = 0.45`
- Источник: `reports/chapter5/hybrid_xiris_summary.json`
- Объект для разбора: `reports/chapter5/hybrid_xiris_blocking_case.json`
- CSV содержит вычисленные поля `chi_R_crit`, `chi_Auto`, `reason`.

BEACON-XAI:

- `total_signals = 100`
- `valid_after_adapter = 83`
- `baseline_manual_checks = 64`
- `fuzzyxai_manual_checks = 11`
- `audit_reports = 12`
- Источник: `reports/chapter5/beacon_xai_summary.json`

GIS INTEGRO:

- `probability = 0.67`
- `mean_alpha_k = 0.72`
- `positive_SHAP_support = 0.47`
- `gamma_route = 0.20`
- `Delta = 0.08`
- Источник: `reports/chapter5/gis_integro_route_metrics.json`

GD-ANFIS/SHAP:

- Фактические значения брать из `reports/chapter5/gd_anfis_shap_report.json`: сейчас `Delta = 0.16`, `I_pre = 0.71`, `action = audit_report`.
- В текущем отчёте фиксируются `n_rules`, `alpha_k`, `eta_k`, `Delta`, `u_k`, `I_pre`, `action`.

## GUI-скриншоты

Папка: `reports/gui_screenshots/`

- `01_home_dashboard.png`
- `02_hybrid_xiris_route.png`
- `03_hybrid_xiris_result.png`
- `04_beacon_audit_route.png`
- `05_beacon_audit_result.png`
- `06_gis_integro_route_report.png`
- `07_gd_anfis_shap_route_report.png`
- `08_evidence_center.png`
- `09_developer_details.png`

## GUI

Запуск:

```bash
bash start_gui.sh
```

URL:

```text
http://localhost:8501
```

Основные экраны:

- Dashboard
- HYBRID-XIRIS
- BEACON-XAI
- GIS INTEGRO
- GD-ANFIS/SHAP
- Evidence Center
- Developer details

Экспорт скриншотов:

```bash
bash export_gui_screenshots.sh
```

GUI берёт данные из `registry/modules.json`, `reports/chapter5/*.json`, `tables/*.md`, `manifest_sha256.json`, `checksums.sha256`.

## GUI acceptance

GUI считается готовым, если пользователь без чтения кода понимает:

- какой сценарий выбран;
- какие данные используются;
- какие параметры заданы;
- какое действие выбрано;
- почему выбрано это действие;
- какие claims разрешены;
- какие claims запрещены;
- где находится evidence.

## Запреты claims

- Не писать, что fixture-маршруты доказывают превосходство исходных моделей.
- Не писать external benchmark там, где указан fixture/source-pending.
- Все числа брать из JSON/CSV, не из текста и не из скриншотов.
