# CLI Usage FuzzyXAI

## Назначение CLI

CLI нужен для запуска FuzzyXAI без написания Python-кода. Он проверен в release-candidate пакете через чистую установку во временной среде.

## Команды

```bash
fuzzyxai validate --payload payload.json --schema classification
fuzzyxai run --payload payload.json --adapter tabular_classification --out out/
fuzzyxai verify --route out/route.json --proof out/proof_trace.json
fuzzyxai render --route out/route.json --out out/dashboard.png
fuzzyxai package --route out/route.json --out out/audit_package.zip
fuzzyxai list-adapters
fuzzyxai list-operators
```

## Проверенный CLI-прогон

В RC-пакете сохранён полный CLI-прогон:

```text
reports/release/fuzzyxai_framework_rc/cli_check/
```

Созданные файлы:

```text
payload.json
route.json
operator_trace.json
operator_table.csv
proof_trace.json
verifier_report.json
dashboard_data.json
dashboard.png
audit_package.zip
cli_check_output.txt
```

Результат:

```text
fuzzyxai validate: PASS
fuzzyxai run: PASS
fuzzyxai verify: PASS
fuzzyxai render: PASS
fuzzyxai package: PASS
```

## Операторный результат CLI

```text
gamma = 0.32
delta = 0.39
rho = 0.39
diagnostic = D_external_tabular_uncertainty
action = lower_confidence
verifier = passed
```

## Registry inspection

Команда `list-adapters` показывает доступные адаптеры. Команда `list-operators` показывает операторы, их входные и выходные контракты, формулы и производимые значения. В RC-пакете эти выводы сохранены в:

```text
reports/release/fuzzyxai_framework_rc/schemas/adapter_registry.json
reports/release/fuzzyxai_framework_rc/schemas/operator_registry.json
```

## Вывод

CLI подтверждает, что FuzzyXAI можно использовать как самостоятельный инструмент: проверить payload, построить маршрут, проверить proof trace, отрисовать dashboard и собрать audit package.
