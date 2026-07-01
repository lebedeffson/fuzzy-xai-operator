# Research Analysis: Sensitivity, Ablation, Cross-Model

## Цель

Research validation показывает переносимость FuzzyXAI на разные задачи. Research analysis дополняет её проверкой поведения операторов: как меняются `gamma`, `delta`, `rho` и action при изменении параметров, отключении компонентов и сравнении моделей.

## Sensitivity analysis

Файл:

```text
research_validation/sensitivity/sensitivity_results.csv
```

Проверенная сетка:

```text
class_probability: 0.55, 0.65, 0.75, 0.90
missing_rate: 0.0, 0.1, 0.2, 0.4
top_k: 10, 5, 3, 1
delta: 0.1, 0.3, 0.5, 0.7
```

Всего:

```text
64 rows
```

Смысл: при росте неопределённости, missing rate или потерь редукции растёт `rho`, а action переходит из `accept` в `lower_confidence` или `audit`.

Фигуры:

```text
assets/rho_surface.png
assets/action_transition_heatmap.png
```

## Ablation study

Файлы:

```text
research_validation/ablation/ablation_summary.csv
research_validation/ablation/ablation_action_changes.csv
```

Проверяются варианты:

```text
full
without_gamma
without_delta
without_quality
without_representation_selection
without_verifier
```

Ключевой результат: отключение `delta` меняет действие в части экспериментов. Например:

```text
wine_gradient_reduced:
baseline_action = lower_confidence
action_without_delta = accept
action_changed = True
```

Это показывает, что оператор редукции влияет на итоговое решение, а не является декоративным числом.

## Cross-model comparison

Файл:

```text
research_validation/cross_model/cross_model_summary.csv
```

Сводка содержит:

```text
model_name
mean_gamma
mean_delta
mean_rho
dominant_action
dominant_risk_component
```

Всего:

```text
12 model families
```

Фигура:

```text
assets/cross_model_mean_rho.png
```

## Дополнительные фигуры research validation

```text
assets/gamma_delta_scatter.png
assets/representation_class_coverage.png
```

`gamma_delta_scatter` показывает положение экспериментов в пространстве рассогласования и потерь объяснения. `representation_class_coverage` показывает активацию классов `F0`, `F_int`, `NAS`, `F_ML`.

## Ограничения

Анализ не утверждает прикладную промышленную или клиническую пригодность. Он проверяет исследовательское свойство framework layer: операторные значения, action и diagnostic меняются при контролируемом изменении входных ограничений.

## Вывод

Sensitivity analysis показывает устойчивую реакцию `rho` и action на параметры. Ablation study показывает вклад отдельных операторов. Cross-model comparison показывает различия между модельными семействами. Вместе эти материалы поддерживают описание FuzzyXAI как исследовательски проверенного фреймворка.
