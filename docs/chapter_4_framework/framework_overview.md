# FuzzyXAI Framework Overview

## Назначение

FuzzyXAI рассматривается как устанавливаемый Python-фреймворк для построения проверяемого объяснительного маршрута. Он не является сайтом DubnaXAI и не является набором прикладных сценариев. Его роль состоит в том, чтобы принять внешний результат модели, привести его к внутреннему контракту `AdaptedInput`, построить операторный маршрут, вычислить ограничения объяснения, выбрать диагностическое состояние и действие, сохранить proof trace и подготовить dashboard.

## Граница фреймворка

Исходный код фреймворка расположен в:

```text
framework/fuzzyxai/
```

Проверенный RC-пакет подтверждает, что импорт идёт из:

```text
framework/fuzzyxai/fuzzyxai/__init__.py
```

а не из старого корневого пакета или из `applications/scenarios`.

## Основные компоненты

| Компонент | Роль |
|---|---|
| `FuzzyXAI` | runtime facade для SDK-запуска |
| `ExplainPlan` | явный план порогов и политик объяснения |
| adapters | преобразование внешнего payload в `AdaptedInput` |
| operator registry | описание операторов, формул и контрактов |
| `OperatorRoute` | вычислительный маршрут |
| `ProofTrace` | доказательный след |
| verifier | проверка согласованности route/proof/action |
| CLI | запуск фреймворка из командной строки |
| dashboard v2 | статическая визуализация готовой трассы |

## Проверенный RC

Проверенный release-candidate пакет:

```text
reports/release/fuzzyxai_framework_rc_package.zip
```

Параметры RC:

```text
source_commit = 1ffcfa97caef6b799959756fca9ad451e72406af
package_type = fuzzyxai_framework_release_candidate
clean_install = PASS
cli_check = PASS
sdk_check = PASS
schema_check = PASS
research_analysis_check = PASS
```

## Вывод

FuzzyXAI оформлен как самостоятельный framework layer: он устанавливается, импортируется, имеет public SDK, CLI, схемы, registry операторов, трассируемый маршрут, verifier и исследовательские проверки. Это делает его инженерной основой главы 4.
