# Visual Analytics Layer of FuzzyXAI

## Идея

FuzzyXAI visual analytics показывает не локальный вклад признаков в предсказание, а трассу формирования доверия к объяснению. Визуальная единица фреймворка:

```text
operator -> component -> risk -> diagnostic -> action -> proof
```

Поэтому визуализация FuzzyXAI отвечает на вопрос:

```text
почему объяснение прошло именно такой операторный маршрут и почему итоговое действие стало таким
```

## Отличие от feature-contribution подхода

В feature-attribution визуализациях основная линия:

```text
feature -> contribution -> prediction
```

В FuzzyXAI основная линия:

```text
input -> explanation object -> representation -> gamma -> delta -> rho -> diagnostic -> action -> proof
```

Это делает визуальный слой пригодным для анализа деградации доверия к объяснению.

## Canonical Visualizations

Введены 8 базовых визуализаций.

| ID | Visualization | Назначение |
|---|---|---|
| V1 | Operator Route Sankey | поток значений между операторами |
| V2 | Gamma-Delta Action Map | геометрия решения в пространстве `gamma`/`delta` |
| V3 | Risk Waterfall | вклад uncertainty/reduction/quality/conflict в `rho` |
| V4 | Operator Trace Heatmap | матрица эксперименты x операторные компоненты |
| V5 | Representation Class Atlas | активация `F0`, `F_int`, `NAS`, `F_ML` |
| V6 | Explanation Coverage Curve | покрытие top-k объяснения и потеря `delta` |
| V7 | Action Boundary Plot | положение `rho` относительно границ action |
| V8 | Proof Consistency Matrix | согласованность route/proof/dashboard/verifier/manifest |

## Single-case visualizations

Для одного внешнего решения используются:

```text
assets/operator_route_sankey.png
assets/risk_waterfall.png
assets/explanation_coverage_curve.png
assets/action_boundary_plot.png
assets/proof_consistency_matrix.png
assets/operator_dashboard_v3.png
```

Пример RC payload:

```text
gamma = 0.32
delta = 0.39
rho = 0.39
dominant_component = delta
action = lower_confidence
```

Смысл: результат не блокируется, но доверие понижено, потому что потеря редуцированного объяснения удерживает `rho` в зоне `lower_confidence`.

## Research visualizations

Для исследовательского набора используются:

```text
assets/gamma_delta_action_map_viz.png
assets/operator_trace_heatmap.png
assets/representation_atlas.png
```

Эти фигуры показывают:

- расположение экспериментов в пространстве `gamma`/`delta`;
- изменение риска и компонент по 20 экспериментам;
- покрытие классов представления по task type и perturbation.

## Dashboard v3

`operator_dashboard_v3` объединяет:

```text
Operator Route Sankey
Risk Waterfall
Action Boundary Plot
Proof Consistency Matrix
```

Dashboard v3 не пересчитывает операторные значения. Он строится из уже сохранённых:

```text
route.json
operator_trace.json
proof_trace.json
verifier_report.json
dashboard_data.json
```

## CLI

В CLI добавлена группа:

```bash
fuzzyxai visualize ...
```

Примеры:

```bash
fuzzyxai visualize route-sankey --route route.json --out operator_route_sankey.png
fuzzyxai visualize risk-waterfall --trace operator_trace.json --out risk_waterfall.png
fuzzyxai visualize gamma-delta-map --results research_validation_results.csv --out gamma_delta_action_map.png
fuzzyxai visualize proof-matrix --package audit_package.zip --out proof_consistency_matrix.png
```

## Release artifact

Проверенный пакет визуализаций:

```text
reports/release/fuzzyxai_visualization_package.zip
```

Он содержит PNG и HTML для single-case и research visual analytics, `visualization_report.md` и `manifest.json` с SHA256.

## Вывод

Визуальный слой FuzzyXAI формирует собственный язык объяснимости: не feature contribution, а operator-risk-action trace. Это поддерживает главную идею главы 4: фреймворк объясняет не только результат модели, но и путь ограничения доверия к объяснению.
