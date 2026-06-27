# FuzzyXAI full pipeline demo

Status: **PASS**

Formula: `D -> F -> T -> M(x) -> XAI(x) -> A(x) -> R(x) -> B;  E_G = E_B ⊙ E_R ⊙ E_A ⊙ E_X ⊙ E_M ⊙ E_T ⊙ E_F ⊙ E_D`

## Case

- risk_score: `0.924851`
- selected class: `FML-audit`
- Delta: `0.196250`
- I(E_G): `0.700974`

## Stages

- **D / Данные**: data_quality=1.000 — проверка полноты и трассируемости входной таблицы
- **F / Data mining / признаки**: top_feature_signal=0.503 — какой признак сильнее всего влияет на риск
- **T / Обучение**: train_accuracy=1.000 — наблюдение качества обучения
- **M / Прогноз модели**: risk_score=0.9249 — перевод прогноза в E_M и термы low/medium/high
- **X / Локальное объяснение**: top_contribution=0.503 — наблюдение вклада признаков
- **A / Решающий модуль**: high risk -> send_to_check — проверка совместимости правила решения с моделью
- **R / Модуль риска**: request_more_data — выбор безопасного действия
- **B / Обратная связь**: case saved — сохранение кейса для калибровки

## Risk observer

- accepted_accuracy: `1.000000`
- coverage: `0.475524`
- defer_rate: `0.000000`
- cost_before: `0.174825`
- cost_after: `0.013636`
- risk_reduction: `0.161189`

## Artifacts

- membership_html: `01_memberships.html`
- feature_html: `02_feature_contributions.html`
- representation_html: `03_representation.html`
- composition_html: `04_composition_graph.html`
- index_html: `reports/full_demo/index.html`
- json: `reports/full_demo/full_pipeline_report.json`
- markdown: `reports/full_demo/full_pipeline_report.md`
