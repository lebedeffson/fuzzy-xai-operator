# FuzzyXAI Experiments

Воспроизводимый слой чисел для глав 2-5. Он не заменяет основное приложение, а собирает проверяемые JSON/CSV для текста диссертации.

## Запуск

```bash
bash run_all.sh
# или
bash fuzzyxai_experiments/run_all.sh
```

## Ключевые отчеты

- `reports/ch2_bc_results.json`: Breast Cancer, калибровка, `I_pre`, `rho`.
- `reports/ch2_synthesis.json`: ограниченный синтез `T_ij`.
- `reports/ch2_critical_ruptures.json`: critical ruptures для главы 2.
- `reports/ch3_selection.json`: выбор класса представления.
- `reports/ch3_reduction.json`: редукции и `Delta` для `sample_113`.
- `reports/ch3_diagnostic_stand.json`: controlled critical ruptures главы 3.
- `reports/ch4_integration.json`: интеграционные статусы и `delta_M`.
- `reports/ch5_hybrid.json`: HYBRID-XIRIS 1000 synthetic degradation cases.
- `reports/ch5_gis.json`: GIS INTEGRO `gamma_route` и `Delta`.
- `reports/ch5_beacon.json`: BEACON-XAI fixture protocol `83/64/11/12`.

## Честность claims

- GIS/BEACON/GD внешние сценарии показывают adapter/report route, а не превосходство исходных моделей.
- Если `delta_M = not measured`, число не выдумывается.
- `ch5_hybrid.json` и `ch5_beacon.json` используют deterministic fixture protocol, не production validation.

## Docker

```bash
docker build -t fuzzyxai-experiments .
docker run --rm fuzzyxai-experiments
```
