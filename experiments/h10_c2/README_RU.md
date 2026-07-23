# H10-C2: предрегистрационный контур

Пакет отдельно проверяет:

- `H10-C2a`: принадлежность найденного диагностического разреза множеству всех
  разрезов минимальной стоимости;
- `H10-C2b`: фактическую повторную сертификацию после выполнения плана
  восстановления в изолированной копии маршрута.

Пакет не изменяет FuzzyXAI v21 и не содержит подтверждающих результатов.
Обычный маршрут завершается статусом `BLOCKED_HUMAN_ADJUDICATION`. Команда
закрытого расчета отделена от остальных целей и требует подписанного разрешения.

```bash
make h10-c2-bootstrap H10_C2_PYTHON=/path/to/existing/python
make h10-c2-test H10_C2_PYTHON=/path/to/existing/python
make h10-c2-power H10_C2_PYTHON=/path/to/existing/python
make h10-c2-generate-development H10_C2_PYTHON=/path/to/existing/python
make h10-c2-run-development H10_C2_PYTHON=/path/to/existing/python
make h10-c2-generate-protocol-validation H10_C2_PYTHON=/path/to/existing/python
make h10-c2-run-protocol-validation H10_C2_PYTHON=/path/to/existing/python
make h10-c2-build-adjudication H10_C2_PYTHON=/path/to/existing/python
make h10-c2-preconfirmatory-gate H10_C2_PYTHON=/path/to/existing/python
```

`make h10-c2-score-sealed` намеренно не входит в обычный маршрут.

