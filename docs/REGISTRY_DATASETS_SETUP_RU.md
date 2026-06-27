# Подготовка малых датасетов registry.cit.gov.ru

## Зачем это нужно
Режимы `registry_*` в FuzzyXAI используются как проверка переносимости контура на разные домены (таблично-текстовый, медицинский аудит, промышленный контроль).

## Быстрая проверка готовности
```bash
make dataset-modes-check
```

Вывод покажет `READY/MISSING/ERROR` по каждому режиму.

## 1) registry_programs
- Карточка: `https://registry.cit.gov.ru/datasets/3e7061cf-3ece-4510-a115-1fc61c369ebf`
- Ожидаемые локальные пути (любой один):
  - `data/cit_registry/registry_programs.csv`
  - `data/cit_registry/programs_registry.csv`
  - `data/cit_registry/registered_programs.csv`
- Минимум: нужен бинарный target-столбец (например, `risk_target`).

## 2) registry_mosmed_doctor_analysis
- Карточка: `https://registry.cit.gov.ru/datasets/c444e643-91be-423f-99bd-abcb63b8f410`
- Ожидаемые локальные пути (любой один):
  - `data/cit_registry/registry_mosmed_doctor_analysis.csv`
  - `data/cit_registry/mosmed_doctor_analysis.csv`
  - `data/cit_registry/mosmed_radiography_doctor_results.csv`
- Минимум: нужен бинарный target-столбец.

## 3) registry_steel_ir
- Карточка: `https://registry.cit.gov.ru/datasets/99eb99d2-cad4-4c47-a017-2ae927018478`
- Ожидаемые локальные пути (любой один):
  - `data/cit_registry/registry_steel_ir.csv`
  - `data/cit_registry/steel_ir_features.csv`
  - `data/cit_registry/industrial_steel_ir.csv`
- Минимум: нужен бинарный target-столбец.

## Формат и подготовка
- Поддерживаются CSV/TSV/XLSX (через общий loader).
- Если в исходнике нет target-столбца, добавьте `risk_target` (0/1).
- Если имена колонок нестандартные, это допустимо: главное наличие бинарного target.

## Важно
Если локальный файл не найден, режим остаётся видимым в GUI как предметный режим, но observer pipeline для него не запускается (статус `MISSING`).

