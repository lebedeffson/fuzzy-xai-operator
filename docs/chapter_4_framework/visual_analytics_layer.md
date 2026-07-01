# Visual Analytics Layer of FuzzyXAI

## Идея

FuzzyXAI visual analytics показывает не локальный вклад признаков в предсказание, а трассу формирования доверия к объяснению:

```text
operator -> risk evidence -> aggregation rule -> rho -> diagnostic -> action -> proof
```

Главное отличие от SHAP: SHAP объясняет `feature -> contribution -> prediction`, а FuzzyXAI объясняет `operator -> risk -> action`.

## Difference from SHAP

SHAP waterfall математически аддитивен:

```text
base value + feature contributions = prediction
```

Для FuzzyXAI это нельзя копировать напрямую, потому что текущий риск чаще агрегируется так:

```text
rho = max(gamma, delta, quality, conflict, interval)
```

Поэтому FuzzyXAI использует SHAP-like readability, но FuzzyXAI-correct semantics. Вместо cumulative waterfall вводится `Local Risk Evidence Bridge`: параллельные evidence bars, линия `rho=max(...)`, доминирующий компонент и action boundary.

## SHAP-like FuzzyXAI Visualizations

| ID | Visualization | Назначение |
|---|---|---|
| S1 | Operator Risk Contribution Summary | среднее значение компоненты, dominance rate и превышение warning-порога |
| S2 | Local Risk Evidence Bridge | локальные evidence bars и `rho=max(...)` без ложного суммирования |
| S3 | Gamma-Delta Action Map v2 | зоны действий по `rho=max(gamma, delta)` и ExplainPlan thresholds |
| S4 | Action Boundary Strip v2 | положение `rho`, расстояния до границ и dominant evidence |
| S5 | Compact Operator Trace Heatmap v2 | числовая матрица риска без смешивания категорий |
| S6 | Representation Class Atlas v2 | `task_type x perturbation -> representation_class` с count |
| S7 | Explanation Coverage Curve | top-k coverage и `delta = 1 - coverage` |
| S8 | Proof Consistency Matrix v2 | artifact x invariant для воспроизводимости |

## Chapter Figures

Основной набор для главы 4:

```text
operator_risk_contribution_summary.png
local_risk_evidence_bridge.png
gamma_delta_action_map_v2.png
action_boundary_strip_v2.png
compact_operator_trace_heatmap_v2.png
representation_class_atlas_v2.png
proof_consistency_matrix_v2.png
explanation_coverage_curve_v2.png
```

Каждая chapter-ready фигура экспортируется в:

```text
PNG
PDF
SVG
```

## Interpretation

`Operator Risk Contribution Summary` не показывает “средний вклад в сумму”. Он показывает:

```text
mean_value
std_value
dominance_count
dominance_rate
mean_excess_over_warning
max_value
```

Это важно для `max`-агрегации: компонент может редко доминировать, но именно он переводит отдельные случаи в `audit` или `defer_to_human`.

`Gamma-Delta Action Map v2` строит фон не вручную, а по правилу:

```text
rho = max(gamma, delta)
```

Поэтому `accept` — это нижний левый квадрат, где одновременно `gamma < rho_accept` и `delta < rho_accept`.

`Action Boundary Strip v2` показывает, почему решение не перешло в соседнюю зону:

```text
rho = 0.39
action = lower_confidence
dominant evidence = reduction
```

## Release Artifact

Проверенный пакет:

```text
reports/release/fuzzyxai_shap_like_visualization_package.zip
```

Он содержит `manifest.json`, `shap_like_visualization_report.md`, PNG/PDF/SVG figure exports и исходные CSV/JSON:

```text
data/operator_risk_contribution_summary.csv
data/local_risk_evidence_bridge.json
data/visual_source_index.json
```

## Вывод

Визуальный слой FuzzyXAI не копирует SHAP математически. Он заимствует читаемость, но сохраняет операторную семантику:

```text
SHAP: additive feature contributions
FuzzyXAI: operator evidence + risk aggregation + action boundary
```
