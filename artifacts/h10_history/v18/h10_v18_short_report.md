# Краткий отчёт H10 v18

## 1. Что реализовано

Реализованы многостадийный типизированный аудитор, иерархическая таксономия из 15 leaf-типов и пяти parent families, dual-threshold open-set detector, отдельные классификатор и локализатор, точный branch-and-bound и приближённый weighted hitting-set solver, repair planner, повторная сертификация и канонический побайтно воспроизводимый trace.

## 2. Независимые baseline

Отдельно реализованы `schema_only`, `hash_version`, `simple_or`, `independent_if_else`, `untyped_graph`, `anomaly_detector` и `typed_route`. AST-тест подтверждает отсутствие импортов `H10Auditor`, `TypedRouteGuard`, `RepairPlanner`, `RepairSetPlanner` и `DiagnosticCutSolver` из baseline.

## 3. Данные

Использованы новые наборы UCI Raisin, UCI Sentiment Labelled Sentences и UCR FordA: 8 804 уникальных объекта и 1 755 sealed route cases. Пересечения train/development/test identities равны нулю. Выполнено одно scoring opening без post-lock tuning.

## 4. Подтверждённые утверждения

Только H10-T: при одинаковых входе, конфигурации и версии новый конвейер сформировал побайтно одинаковый audit trace для всех 1 755 повторных проверок.

## 5. Неподтверждённые или недействительные утверждения

H10-L, H10-C и H10-R не могут использоваться как confirmatory claims. Closure-аудит установил, что ground truth диагностического разреза создавался тем же solver, который оценивался у `full_h10`, а source truth и локализатор использовали общую taxonomy mapping. Исходные статистические результаты сохранены, но их положительные автоматические статусы аннулированы. H10-U остаётся описательной.

## 6. Ограничения

Контролируемые мутации не заменяют естественные production faults. Сравнительная проверка требует независимого adjudication oracle. Три набора меньше exploratory power estimate в пять наборов. Predictive labels не являются целью H10.

## 7. Отклонения

Replay признан недействительным: normal stream был построен из уже мутированных sealed routes, что вызвало ложные предупреждения около 8 991 на 10 000 событий. Replay не пересчитывался после обнаружения ошибки. Старые и исходные replay-файлы сохранены с префиксом `invalid_original_`. Sealed scoring не повторялся.

## 8. Воспроизведение

```bash
make H10_PYTHON=/home/lebedeffson/Code/venv/bin/python reproduce-h10
```

Команда проверяет тесты и пересобирает производные артефакты из уже завершённого scoring. Она не открывает label vault повторно. Для нового сравнительного claim требуется свежий protocol и новые sealed data.
