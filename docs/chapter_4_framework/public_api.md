# Public API FuzzyXAI

## Установка

```bash
pip install -e framework/fuzzyxai
```

## Базовые импорты

```python
from fuzzyxai import FuzzyXAI, ExplainPlan
from fuzzyxai.adapters import get_adapter
```

Также остаются доступны функциональные входы:

```python
from fuzzyxai import (
    build_explainable_object,
    build_route,
    build_proof_trace,
    verify_proof_trace,
    render_dashboard,
    save_route_json,
)
```

## Runtime SDK

Минимальный пользовательский сценарий:

```python
from fuzzyxai import FuzzyXAI, ExplainPlan
from fuzzyxai.adapters import get_adapter

payload = {
    "scenario_id": "external_wine_classification",
    "source_type": "tabular",
    "model_name": "ExternalDemoModel",
    "dataset_name": "manual_payload",
    "predicted_class": 1,
    "class_probability": 0.68,
    "feature_values": {"x1": 0.4, "x2": 0.7, "x3": 0.2},
    "feature_importance": {"x1": 0.31, "x2": 0.18, "x3": 0.12},
    "quality_metrics": {"missing_rate": 0.05, "feature_range_violation": 0.0},
    "context_values": {"external_task": True},
}

fxai = FuzzyXAI(plan=ExplainPlan.default())
adapter = get_adapter("tabular_classification")

route = fxai.run_payload(payload, adapter)
report = fxai.verify(route)
fxai.export_package(route, "out")

print(route.computed_result)
print(report.valid)
```

Проверенный RC-результат:

```text
gamma = 0.32
delta = 0.39
rho = 0.39
diagnostic = D_external_tabular_uncertainty
action = lower_confidence
verifier = passed
```

## ExplainPlan

`ExplainPlan` фиксирует параметры интерпретации:

| Поле | Назначение |
|---|---|
| `gamma_warning`, `gamma_critical` | пороги рассогласования |
| `delta_warning`, `delta_critical` | пороги потери объяснения |
| `rho_accept`, `rho_warning`, `rho_audit`, `rho_critical` | зоны риска |
| `top_k` | размер редуцированного объяснения |
| `representation_policy` | политика выбора класса представления |
| `action_policy` | политика выбора действия |

## Adapter SDK

Адаптеры регистрируются через registry и должны поддерживать:

```text
adapter_id
task_type
supported_payload_schema
validate_payload(payload)
to_adapted_input(payload)
describe()
```

Текущий проверенный внешний адаптер:

```text
tabular_classification
```

## Вывод

Public API позволяет использовать FuzzyXAI как библиотеку: внешний пользователь создаёт payload, выбирает adapter, запускает `FuzzyXAI`, получает route, proof trace, dashboard и audit package.
