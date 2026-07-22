# FuzzyXAI H10 v19 — итоговый отчёт по исправленной практике

## Статус

Сравнительный цикл H10 v19 завершён с независимым adjudication oracle.
Методологический аудит: **PASS**. Sealed scoring выполнен один раз.
Post-lock tuning отсутствует. Старые v16/v18 результаты не изменялись.

## Исправления относительно недействительного v18

- Эталонные source nodes и repair targets выводятся из независимого mutation log.
- Oracle не импортирует H10Auditor, taxonomy, SourceLocalizer, DiagnosticCutSolver или baseline.
- Все равностоимостные оптимальные diagnostic cuts учитываются как допустимые.
- Replay использует только clean routes; мутированные маршруты не входят в normal stream.
- Времена восстановления не задаются отдельно для методов.
- Baseline import independence и oracle independence проверяются автоматически.

## Confirmatory-результаты

### H10-L: локализация источника

- Full H10 macro-F1: **0,86855**.
- Лучший независимый baseline: **0,81883**.
- Иерархический эффект H10 minus baseline: **+0,04941**.
- 95% CI: **[0,02591; 0,07251]**.
- Holm p: **0,01440**.
- Направление положительно на всех трёх наборах.
- Статус: **supported**.

### H10-R: repair set

- Full H10 F1: **0,86855**.
- Лучший независимый baseline: **0,81883**.
- Иерархический эффект: **+0,04941**.
- 95% CI: **[0,02647; 0,07154]**.
- Holm p: **0,01440**.
- Направление положительно на всех трёх наборах.
- Статус: **supported**.

### H10-C: minimal diagnostic cut

Результат является вторичным описательным:

- Exact match full H10: **0,99873**.
- Exact match independent if-else: **0,46422**.
- Jaccard full H10: **0,99937**.
- Cost ratio full H10: **1,00152**.
- Cost ratio independent if-else: **1,63580**.

### H10-U: unknown faults

Только описательный результат:

- Unknown AUROC: **0,71471**.
- Unknown recall: **0,36563**.

Open-set detection не следует представлять как закрытый сильный результат.

### H10-T: воспроизводимость trace

- Побайтная идентичность audit trace: **1,0** на 1 926 повторных проверках.
- Статус: **supported**.

## Safety point estimates

- False certification full H10: **0,0**.
- False block full H10: **0,0**.
- Зарегистрированные границы 0,01 выполнены по точечной оценке.

## Replay

Управляемый replay содержит 1 000 000 событий и 60 инцидентов.

Для full H10:

- incident recall: **1,0**;
- false alerts на 10 000 clean events: **0,0**;
- recertification rate: **1,0**;
- repair success: **0,95**;
- manual load: **0,2833**;
- erroneous repair actions: **5**.

Replay остаётся контролируемой симуляцией и не является production validation.

## Ограничения

- Использованы те же публичные identity anchors, что и в недействительном v18, но с новым salted split, новыми opaque case IDs, новым mutation schedule, новым vault и независимым oracle.
- Route metadata и faults контролируемые, а не снятые с производственной системы.
- H10-U остаётся слабым описательным результатом.
- Результаты не подтверждают улучшение прогнозных решений и не изменяют отрицательную H3.

## Финальный вывод

Практика H10 исправлена методологически. Подтверждено умеренное, но статистически поддержанное преимущество типизированного аудитора над сильной независимо реализованной rule-oriented схемой по локализации источника и качеству repair set. Наиболее сильный технический результат minimal cut остаётся вторичным, а open-set detection требует дальнейшей работы.
