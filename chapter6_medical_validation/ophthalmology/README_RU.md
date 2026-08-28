# Глава 6 — офтальмология: PAPILA, глаукома и LIME

Это исследовательский, не устанавливаемый с wheel контур применения
зафиксированного FuzzyXAI P19 к binary классификации fundus ROI: healthy vs glaucoma.
Эксперимент является **воспроизводимым продолжением опубликованной постановки
на открытых данных**, а не точной репликацией исходных весов и не клинической
валидацией.

## Текущий статус

- PAPILA v2 получен из официального Figshare API и верифицирован вне Git:
  244 пациента, 488 изображений; raw data и clinical metadata не входят в repository или bundle.
- Primary protocol использует только clean healthy/glaucoma patients в patient-level five-fold CV.
  Любой suspect-associated patient исключён из primary CV и сохраняется для ambiguity cohort.
- ResNet50 с expert-1 optic-disc ROI, LIME и Grad-CAM уже имеют real public
  FuzzyXAI case outputs. См. `CH6_EYE_RESULTS_RU.md`.
- FuzzyXAI core не изменяется; научные поля после `explain_one()` вычисляет
  только frozen public runtime.

## Данные

Задайте корень без сохранения локального пути в artifacts:

```bash
export FUZZYXAI_CH6_DATA_ROOT=/path/to/ch6-eye-data
```

Основной протокол использует PAPILA (official Figshare article 14798004, version 2).
Expert-1 optic-disc contour используется исключительно как детерминированный offline ROI;
expert-2 — только diagnostic annotation variability channel. Clinical metadata не classifier input.
IDRiD и APTOS остаются historical scaffolds, а не исполненными PAPILA results.

После ручного размещения:

```bash
python chapter6_medical_validation/ophthalmology/scripts/download_papila.py --extract
python chapter6_medical_validation/ophthalmology/scripts/verify_papila.py
python chapter6_medical_validation/ophthalmology/scripts/freeze_papila_cv.py
```

Verifier сначала проверяет official payload, label/image linkage, hashes,
segmentation completeness и dimensions. Затем CV script один раз создаёт
immutable patient-level manifest; он не перезаписывает отличающийся content.

Дальнейший порядок фиксирован: outer folds, internal-validation model selection,
canonical fold 5 case selection, LIME + Grad-CAM, затем public `explain_one`.
Case script после публичного вызова экспортирует `ModelExplanationResult`; Gamma,
uncertainty, I_pre, strict rho и action повторно в experiment code не вычисляются.

CUDA/torchvision runtime исправлен в отдельном overlay-venv; текущий blocker
относится только к официальному доступу к IDRiD.

## Научная граница

Метрики классификатора не приписываются FuzzyXAI. Пространственное совпадение
Grad-CAM/IG с lesion masks — диагностическая проверка, не причинность и не
клиническая корректность. Пороги риска — технический профиль ExplainPlan, не
клинические пороги. Сырые изображения, masks и checkpoints в Git/ZIP не входят.
