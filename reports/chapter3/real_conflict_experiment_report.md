# Эксперимент на реальных объяснительных конфликтах

Датасет: Breast Cancer Wisconsin. Это не клиническая апробация, а проверка XAI-конфликтов на реальной табличной задаче.

- Уникальных объектов в датасете: 569
- Уникальных объектов в evaluation cases: 227
- Evaluation cases: 1000
- Формирование 1000 cases: bootstrap из validation/test split; это evaluation cases, а не уникальные клинические объекты.
- Модель: logistic regression, одна и та же модель для SHAP и LIME.
- Test accuracy модели: 0.9649
- Реальный конфликт: rank/sign/rule расхождение между SHAP, LIME и rule/ExplainPlan-профилем.

| Метрика | Значение |
| --- | ---: |
| `n_cases` | 1000 |
| `n_unique_objects` | 227 |
| `n_conflicts` | 938 |
| `n_critical_conflicts` | 917 |
| `rank conflicts` | 697 |
| `sign conflicts` | 615 |
| `rule conflicts` | 385 |

Синтетические перевороты правил не используются.
