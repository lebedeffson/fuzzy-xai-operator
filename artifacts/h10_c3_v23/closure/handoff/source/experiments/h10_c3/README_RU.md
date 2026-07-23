# H10-C3 v23

Предподтверждающий контур для двух новых гипотез:

- `H10-C3a`: оптимальный диагностический разрез;
- `H10-C3b`: восстановление и полная повторная сертификация.

S1 является контрольной группой. Основная популяция заранее ограничена S2-S5
и составляет 90% открытых выборок. Development используется для выбора сильнейшего
baseline. После `freeze` код, конфигурация и выбранный baseline хешируются; protocol
validation прекращается при любом post-lock изменении.

Обычный конвейер никогда не создаёт и не открывает sealed-часть:

```bash
make diagnostic-v23-test
make h10-c3-generate-development
make h10-c3-run-development
make h10-c3-freeze
make h10-c3-generate-protocol-validation
make h10-c3-run-protocol-validation
make h10-c3-stability-analysis
make h10-c3-power
make h10-c3-preconfirmatory-gate
make h10-c3-reports
```

`make h10-c3-score-sealed` всегда завершается fail-closed, пока отдельный sealed
цикл не создан после успешного гейта. Результаты открытых частей не являются
confirmatory evidence и не изменяют прежние H3, H5-P, H6-general, H10-L или H10-R.

