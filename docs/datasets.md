# Datasets for FuzzyXAI Benchmark

Ниже — текущий рабочий набор датасетов для практической проверки контура.

| key | domain | source | license | target | notes |
|---|---|---|---|---|---|
| `breast_cancer` | medicine | sklearn/UCI | UCI terms (via sklearn) | `risk_target` | базовый reproducible medical benchmark |
| `rikord` | medicine | local/public RU medical | depends on source | inferred | при отсутствии локального файла используется fallback-proxy |
| `ruccod` | medicine+text | local/public RU medical | depends on source | inferred | текст+кодирование, fallback-proxy доступен |
| `citr_mosmed` | medical_audit | registry.cit.gov.ru | see card | inferred | режим для аудита врач/модель, fallback при отсутствии файла |
| `citr_steel_ir` | industrial | registry.cit.gov.ru | see card | inferred | промышленный сценарий, fallback при отсутствии файла |
| `synthetic_fraud` | finance | generated | generated | `risk_target` | контролируемый стресс-тест политики риска |

## Команды

Проверка доступности/статусов:

```bash
make dataset-modes-check
```

Бенчмарк одного датасета:

```bash
make benchmark-dataset DATASET=breast_cancer
make benchmark-dataset DATASET=rikord
```

Артефакты:

- `reports/dataset_benchmark/<dataset>/dataset_metadata.json`
- `reports/dataset_benchmark/<dataset>/metrics.json`
- `reports/dataset_benchmark/<dataset>/metrics.csv`
- `reports/dataset_benchmark/<dataset>/dataset_observer_report.{json,md,html}`

