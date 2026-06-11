# Evidence-пакет глав 4-5: FuzzyXAI Studio

## Репозиторий и версия

- Репозиторий: https://github.com/lebedeffson/fuzzy-xai-operator
- Ветка: `feature/math-aligned-code`
- Commit hash сборки: см. `git log --oneline -1` в полученном репозитории; пакет запушен отдельным финальным коммитом
- Дата сборки: `2026-06-11T06:59:39.501803+00:00`

## Запуск с нуля

```bash
bash run_all.sh
python fuzzyxai_experiments/compare_reports.py
```

Ожидаемый итог:

```text
PASS: hybrid_xiris
PASS: beacon_xai
PASS: gis_integro
PASS: gd_anfis_shap
PASS: generated_tables
PASS: checksums
```

## Docker

```bash
docker build -t fuzzyxai-experiments .
docker run --rm fuzzyxai-experiments
```

Примечание: в текущей локальной среде Docker daemon недоступен, поэтому проверен shell-запуск `bash run_all.sh`.

## Структура

- `data/raw/`: примечание по внешним raw-данным.
- `data/fixtures/`: GIS и GD-ANFIS/SHAP fixture-входы.
- `data/generated/`: object-level HYBRID/BEACON данные.
- `reports/chapter4/`: отчёты главы 4.
- `reports/chapter5/`: отчёты главы 5.
- `tables/`: generated tables для диссертации.
- `logs/`: логи запуска и проверки.
- `scripts/`: скрипты baseline/metrics/manifest.
- `registry/modules.json`: реестр сценариев.
- `manifest_sha256.json`, `checksums.sha256`: контрольные суммы.

## HYBRID-XIRIS

- Object-level CSV: `data/generated/hybrid_xiris_objects.csv`
- Summary: `reports/chapter5/hybrid_xiris_summary.json`
- Поля: `object_id, model_score, quality_score, is_critical, baseline_action, fuzzyxai_action`
- Baseline: `accept if model_score > 0.7`
- Числа: всего 1000; critical 168; baseline missed 168; FuzzyXAI missed 0.

## BEACON-XAI

- Input CSV: `data/generated/beacon_xai_signals.csv`
- Summary: `reports/chapter5/beacon_xai_summary.json`
- Adapter failures: `reports/chapter5/beacon_xai_adapter_failures.csv`
- Числа: всего 100; valid after adapter 83; baseline manual checks 64; FuzzyXAI manual checks 11; audit reports 12.

## GIS INTEGRO

- Fixture: `data/fixtures/gis_integro_fixture.csv`
- Summary: `reports/chapter5/gis_integro_route_metrics.json`
- Числа: probability 0.67; mean_alpha_k 0.72; positive_SHAP_support 0.47; gamma_route 0.20; Delta 0.08; status source-pending.
- Claim scope: контрольный маршрут, качество исходной GIS-модели не заявляется.

## GD-ANFIS/SHAP

- Rules: `data/fixtures/gd_anfis_rules.csv`
- SHAP: `data/fixtures/gd_anfis_shap_values.csv`
- Summary: `reports/chapter5/gd_anfis_shap_report.json`
- Метрики: число правил, `alpha_k`, `eta_k`, `Delta`, `u_k`, `I_pre`, action.

## Таблицы для диссертации

- `tables/generated_tables.tex`
- `reports/chapter5/*summary.json`
- `reports/chapter5/*metrics.json`
- `reports/chapter5/gd_anfis_shap_report.json`

## Запреты claims

- Не писать, что fixture-маршруты доказывают превосходство исходных моделей.
- Не писать external benchmark там, где указан fixture/source-pending.
- Все числа брать из JSON/CSV, не из текста и не из скриншотов.
