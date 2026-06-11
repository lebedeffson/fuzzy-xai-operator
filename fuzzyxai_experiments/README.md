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

## GUI

```bash
cd fuzzyxai_experiments
bash run_chapter4_5.sh
python compare_reports.py
bash start_gui.sh
```

URL: `http://localhost:8501`

GUI читает реальные файлы evidence-пакета, а не hardcoded текст. Если отчёты не созданы, dashboard показывает `Evidence not generated` и предлагает запустить pipeline.

Экспорт скриншотов:

```bash
bash export_gui_screenshots.sh
```

Папка: `reports/gui_screenshots/`.

## Ключевые файлы

- `registry/modules.json`: реестр сценариев.
- `data/generated/hybrid_xiris_objects.csv`: 1000 объектов HYBRID-XIRIS с `chi_R_crit`, `chi_Auto`, `reason`.
- `reports/chapter5/hybrid_xiris_blocking_case.json`: один вычисленный blocking case для разбора в главе 5.
- `data/generated/beacon_xai_signals.csv`: 100 сигналов BEACON-XAI.
- `data/fixtures/gis_integro_fixture.csv`: входы для `gamma_route` и `Delta`.
- `data/fixtures/gd_anfis_rules.csv`, `data/fixtures/gd_anfis_shap_values.csv`: правила и SHAP.
- `reports/chapter5/*`: JSON/CSV-отчёты глав 4-5.
- `tables/*.md`, `tables/generated_tables.tex`: таблицы для диссертации.
- `reports/gui_screenshots/*.png`: скриншоты GUI-витрины.
- `gui_app.py`, `start_gui.sh`, `export_gui_screenshots.sh`: локальный GUI FuzzyXAI Studio.
- `manifest_sha256.json`, `checksums.sha256`: контрольные суммы.

## Claims

- Разрешено: маршрут выполнен, отчёт сформирован, метрики воспроизведены.
- Запрещено: заявлять новую точность внешней модели, clinical effectiveness или production-ready статус без отдельной валидации.

## Зафиксированные пороги HYBRID-XIRIS

- `score_threshold = 0.70`
- `quality_threshold = 0.45`
- Baseline: `accept`, если `model_score > 0.70`.
- FuzzyXAI блокирует критический случай при высокой уверенности и низком качестве: `model_score > 0.70` и `quality_score < 0.45`.
