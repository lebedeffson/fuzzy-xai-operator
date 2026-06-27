# Пакет скринов и вставок для глав 4-5

## Как использовать

- Основной путь: `reports/chapter4_5_screens_pack/`.
- Архив: `reports/chapter4_5_screens_pack.zip`.
- Все числа брать только из CSV/JSON внутри `chapter4/tables` и `chapter5/tables`.
- GUI-скрины из `gui_all_screens` лучше ставить в приложение; в основную главу брать только 2-3 самых читаемых.

## Глава 4: рекомендуемые изображения

1. `chapter4/screens/fig_4_evidence_contract.png`

Подпись: Evidence-вкладка FuzzyXAI Studio: проверка ExplainPlan, sample_113 и связанных артефактов.

2. `chapter4/screens/fig_4_ecosystem_registry.png`

Подпись: Интерактивный реестр внешних модулей FuzzyXAI: источник, evidence level, статус и граница утверждения.

3. `chapter4/screens/fig_4_evidence_matrix_heatmap.png`

Подпись: Evidence matrix электронного пакета FuzzyXAI: наличие адаптера, отчёта, фигуры и GUI-видимости для каждого модуля.

4. `chapter4/screens/13_sdk_api_route.png`

Подпись: Открытый SDK/API-маршрут подключения внешнего модуля к FuzzyXAI.

## Глава 4: таблицы

- `chapter4/tables/table_4_evidence_matrix.csv`
- `chapter4/tables/table_4_registry_snapshot.csv`
- `chapter4/tables/table_4_integration_effort_measurements.csv`
- `chapter4/tables/integration_effort_summary.json`

Формулировка: протокол трудоёмкости является протоколом отдельных интеграционных действий, а не production estimate.

## Глава 5: рекомендуемые изображения

1. `chapter5/screens/fig_5_scenario_action_routes.png`

Подпись: Сценарные маршруты прикладных модулей через FuzzyXAI: внешний артефакт преобразуется адаптером в объяснительный объект, диагностическое состояние или отчёт.

2. `chapter5/screens/fig_5_module_channel_coverage.png`

Подпись: Покрытие каналов объяснительного объекта по внешним модулям: `L_k`, `mu_k`, `R_k`, `alpha_k`, `u_k`, `tau_k`, `eta_k`, `D_k`, Report, Action.

3. `chapter5/screens/fig_5_hybrid_xiris_blocking_case.png`

Подпись: Контрольный пример HYBRID-XIRIS: низкое качество сегментации и конфликт источников приводят к `chi_R_crit=1`, `chi_Auto=false`, `action=block`.

4. `chapter5/screens/gis_integro_route.png`

Подпись: GIS INTEGRO как контрольный маршрут через GD-ANFIS/SHAP-каналы; route metrics фиксируются отдельно и не являются сравнением качества исходной модели.

5. `chapter5/screens/beacon_xai_route.png`

Подпись: BEACON-XAI fixture-маршрут с закреплённым репозиторием и commit hash; сценарий подтверждает adapter/report route.

## Глава 5: ключевые числа

GIS INTEGRO:

- `gamma_route = 0.20`
- `Delta = 0.08`
- источник: `chapter5/tables/gis_integro_route_metrics.json`

BEACON-XAI:

- `route_supported = 83/100`
- `baseline_alerts_before = 64`
- `fuzzyxai_alerts_after = 11`
- `audit_reports = 12`
- источник: `chapter5/tables/ch5_beacon.json`

HYBRID-XIRIS:

- `n_images = 1000`
- `baseline_missed_critical_quality_cases = 168`
- `fuzzyxai_missed_critical_quality_cases = 0`
- источник: `chapter5/tables/ch5_hybrid.json`

## Ограничения claims

- Не писать, что GIS INTEGRO/BEACON-XAI/HYBRID-XIRIS доказывают превосходство исходных внешних моделей.
- Писать: воспроизводимо показан adapter/report route, route metrics и safety-действие на контрольных данных.
- Для сценариев с `N/A` или `not_available` не вводить искусственные accuracy/missed critical ruptures.
