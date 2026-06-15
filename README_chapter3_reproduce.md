# Воспроизведение пакета главы 3

Запуск:

```bash
make chapter3-final-evidence
```

Пайплайн:

1. аудит DOCX на зависимости от `F_H`;
2. построение реальных SHAP/LIME/rule конфликтов на Breast Cancer Wisconsin;
3. сравнение режимов `F0`, `NAS`, `F_ML`;
4. автоматическая калибровка весов и порогов по validation split;
5. bootstrap CI 95% и Markdown-вставки для главы;
6. валидация и сборка `chapter3_final_fix_evidence_package.zip`.

Ограничение: Breast Cancer Wisconsin используется как реальная табличная XAI-задача, а не как клиническая апробация. Если исходный DOCX `/mnt/data/03_glava_3_final_reviewer_fixed_submission_ready.docx` отсутствует, аудит создаёт отчёт с явным статусом `source_missing`.
