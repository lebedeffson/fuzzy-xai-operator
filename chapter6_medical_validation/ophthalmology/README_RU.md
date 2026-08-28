# Глава 6 — офтальмология: диабетическая ретинопатия

Это исследовательский, не устанавливаемый с wheel контур применения
зафиксированного FuzzyXAI P19 к пятиступенчатой классификации фундус-снимков.
Эксперимент является **воспроизводимым продолжением опубликованной постановки
на открытых данных**, а не точной репликацией исходных весов и не клинической
валидацией.

## Текущий статус

- Протокол, конфигурации, loaders, leakage checks, preprocessing, model/XAI
  helpers, thin evidence adapter и тестовые fixtures подготовлены.
- Реальный IDRiD в рабочей среде отсутствует; официальный IEEE DataPort route
  требует интерактивного входа/принятия условий.
- Финальные split, checkpoints, метрики и medical case artifacts поэтому имеют
  статус `MISSING_DATA`, а не заполняются синтетическими числами.
- FuzzyXAI core не изменяется; научные поля после `explain_one()` вычисляет
  только frozen public runtime.

## Данные

Задайте корень без сохранения локального пути в artifacts:

```bash
export FUZZYXAI_CH6_DATA_ROOT=/path/to/ch6-eye-data
```

Основной протокол главы 6 использует IDRiD: официальный grading train (413)
детерминированно делится на train/validation, а официальный grading test (103)
сохраняется как final test. Pixel-level lesion masks используются только для
пространственной диагностики XAI и не являются входом классификатора.
Ожидаемая структура и официальный источник описаны в
`configs/dataset_idrid.yaml` и `DATA_ACCESS.md`; credentials в репозиторий не
добавляются. APTOS-конфигурация сохранена только как прежний research scaffold
и не является primary dataset этого протокола.

После ручного размещения:

```bash
python -m chapter6_medical_validation.ophthalmology.scripts.prepare_datasets
python -m pytest -q chapter6_medical_validation/ophthalmology/tests
```

`prepare_datasets` сначала проверяет число объектов, labels, пути и official
IDRiD split. Только затем он один раз создаёт
`outputs/manifests/split_aptos_seed2026.json`. Существующий split без флага
`--verify-only` не перезаписывается.

Дальнейший порядок фиксирован: три запуска `train_classifier` для seeds
2026/2027/2028, отдельная оценка и validation-only temperature scaling,
алгоритмический `select_cases`, `generate_native_xai`, затем
`run_fuzzyxai`. Последний скрипт после публичного `explain_one()` только
экспортирует `ModelExplanationResult`; Γ, неопределённость, I_pre, strict rho
и action повторно в experiment code не вычисляются.

CUDA/torchvision runtime исправлен в отдельном overlay-venv; текущий blocker
относится только к официальному доступу к IDRiD.

## Научная граница

Метрики классификатора не приписываются FuzzyXAI. Пространственное совпадение
Grad-CAM/IG с lesion masks — диагностическая проверка, не причинность и не
клиническая корректность. Пороги риска — технический профиль ExplainPlan, не
клинические пороги. Сырые изображения, masks и checkpoints в Git/ZIP не входят.
