# H10-C3 R4: готовность к подтверждающему запуску

Контур R4 устраняет три ограничения прежнего открытого эксперимента:

- development, protocol validation и sealed используют непересекающиеся
  структурные банки;
- шесть семейств имеют разные графы и контракты;
- H10-C3b выполняет реальное изменение `RouteGraph` через
  `RepairExecutor` и полный `RouteRecertifier`.

Старые результаты v23/v23.1 сохраняются как
`FORMATIVE_SYNTHETIC_RESULT`. Результаты R4 development и protocol validation
остаются открытыми предрегистрационными результатами. Они не являются
confirmatory evidence.

Порядок запуска:

```bash
make h10-c3-r4-test
make h10-c3-r4-template-audit
make h10-c3-r4-generate-development
make h10-c3-r4-run-development
make h10-c3-r4-freeze
make h10-c3-r4-generate-protocol-validation
make h10-c3-r4-run-protocol-validation
make h10-c3-r4-stability
make h10-c3-r4-power
make h10-c3-r4-preconfirmatory-gate
```

После успешного гейта sealed создаётся отдельно:

```bash
make h10-c3-r4-generate-sealed
```

Обычная сборка не запускает scoring. Однократный scoring требует отдельного
разрешающего JSON, совпадающего с protocol lock:

```bash
make h10-c3-r4-score-sealed APPROVAL=/path/to/approval.json
```

До такого разрешения корректные статусы:

```text
H10-C3a: NOT_EVALUATED_CONFIRMATORY
H10-C3b: NOT_EVALUATED_CONFIRMATORY
sealed opening count: 0
```

Человеческие факторы не оцениваются. Ручное согласование не требуется только
для формально определённой алгоритмической области.
