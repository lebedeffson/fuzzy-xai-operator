# FuzzyXAI Experiments Evidence Pack

Воспроизводимый пакет для глав 4-5. Он хранит входные fixture/generated данные, запускаемые сценарии, таблицы, логи, GUI-скриншоты и SHA256-манифест.

## Проверка глав 4-5

```bash
cd fuzzyxai_experiments
bash run_chapter4_5.sh
python compare_reports.py
sha256sum -c checksums.sha256
```

Ожидаемый вывод:

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

## Полный запуск 2-5

```bash
bash run_all.sh
```

Для финализации глав 4-5 использовать именно `run_chapter4_5.sh`.

## Docker

```bash
cd fuzzyxai_experiments
docker build -t fuzzyxai/evidence:chapter4-5 .
docker run --rm fuzzyxai/evidence:chapter4-5 bash run_chapter4_5.sh
```

## Ключевые файлы

- `registry/modules.json`: реестр сценариев.
- `data/generated/hybrid_xiris_objects.csv`: 1000 объектов HYBRID-XIRIS.
- `data/generated/beacon_xai_signals.csv`: 100 сигналов BEACON-XAI.
- `data/fixtures/gis_integro_fixture.csv`: входы для `gamma_route` и `Delta`.
- `data/fixtures/gd_anfis_rules.csv`, `data/fixtures/gd_anfis_shap_values.csv`: правила и SHAP.
- `reports/chapter5/*`: JSON/CSV-отчёты глав 4-5.
- `tables/*.md`, `tables/generated_tables.tex`: таблицы для диссертации.
- `reports/gui_screenshots/*.png`: скриншоты GUI-витрины.
- `manifest_sha256.json`, `checksums.sha256`: контрольные суммы.

## Claims

- Разрешено: маршрут выполнен, отчёт сформирован, метрики воспроизведены.
- Запрещено: заявлять новую точность внешней модели, clinical effectiveness или production-ready статус без отдельной валидации.
