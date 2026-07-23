# API диагностики

## Одиночный маршрут

```python
report = FuzzyXAI().diagnose(
    route=route,
    repair_mode="plan",
    audience="user",
)
```

Поддерживаются `OperatorRoute`, `ProofTrace`, прежний `RouteObservation`, `RouteGraph`
и JSON-compatible mapping.

## Результат объяснения

```python
explanation = FuzzyXAI.wrap(model).explain(objects)
report = explanation.diagnose()
```

## Пакет

```python
batch = FuzzyXAI().diagnose_batch(routes=routes)
```

В batch execution внешних исправлений запрещён.

## CLI

```bash
fuzzyxai diagnose --route route.json --repair-plan \
  --output report.json --html report.html
fuzzyxai diagnose-batch --input routes.jsonl --repair-plan --output reports/
fuzzyxai recertify --before before.json --after after.json \
  --plan plan.json --output recertification.json
```
