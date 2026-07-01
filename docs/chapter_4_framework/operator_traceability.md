# Operator Traceability

## Принцип трассируемости

Операторный маршрут FuzzyXAI должен быть проверяем без чтения исходного кода. Для этого каждый узел маршрута содержит входы, формулу, компоненты расчёта, выходы, пороги, статус, интерпретацию и связи со следующими узлами.

## Узлы маршрута

Типовой внешний маршрут содержит 9 узлов:

```text
input_artifact
explanation_object
representation
alignment
reduction
risk
diagnostics
action
proof
```

Для каждого узла сохраняются:

```text
input_values
output_values
formula_text
components
thresholds
status
status_reason_ru
interpretation_ru
next_node_ids
```

## Рёбра маршрута

`OperatorEdge` фиксирует, что именно передано между операторами:

```text
source_node_id
target_node_id
passed_values
explanation_ru
```

Это позволяет проверить, что `gamma`, `delta`, `rho`, diagnostic и action не появляются в dashboard вручную, а передаются по маршруту.

## Пример вычисления

Для внешнего табличного payload в RC-проверке:

```text
class_probability = 0.68
uncertainty = 1 - 0.68 = 0.32
quality_penalty = 0.05
gamma = max(0.32, 0.05) = 0.32
top_k_importance_sum = 0.61
delta = 1 - 0.61 = 0.39
rho = max(0.32, 0.39) = 0.39
action = lower_confidence
```

Главный вклад в риск даёт `delta`, то есть потеря редуцированного объяснения.

## Trace package

Каждый трассируемый запуск сохраняет:

```text
route.json
operator_trace.json
operator_table.csv
proof_trace.json
verifier_report.json
dashboard_data.json
operator_dashboard_v2.png
operator_dashboard_v2.html
operator_cards/
```

## Dashboard v2

Иллюстрация для главы:

```text
docs/chapter_4_framework/assets/operator_dashboard_v2.png
```

Dashboard v2 строится из `route` и `dashboard_data`; он не пересчитывает `gamma`, `delta`, `rho` и не зависит от сайта DubnaXAI.

## Verifier

Verifier проверяет согласованность:

```text
source_commit_present
route_proof_values_match
action_matches_route
diagnostic_present
operator_nodes_have_io
operator_edges_have_passed_values
gamma_matches_formula
rho_matches_formula
action_matches_risk_zone
```

## Вывод

Трассируемость делает FuzzyXAI не просто генератором картинок, а проверяемым вычислительным фреймворком: каждый итоговый action восстанавливается через входы, формулы, компоненты и proof trace.
