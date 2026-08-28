# Требуется официальный доступ к IDRiD

Статус на 2026-08-28: `MISSING_DATA`.

Официальная страница Grand Challenge перенаправляет загрузку IDRiD в IEEE
DataPort. Автоматическая проверка 2026-08-28 получила HTTP 403 от страницы
загрузки: требуется интерактивный вход/принятие условий, которых нет в
автономной среде. Неофициальные mirrors не используются.

## Единственное действие для получения данных

1. Откройте официальную страницу [IDRiD Data](https://idrid.grand-challenge.org/Data/),
   войдите в IEEE DataPort и примите условия доступа.
2. Скачайте из official repository **Indian Diabetic Retinopathy Image Dataset
   (IDRiD), Complete Data** — grading images/CSV и pixel-level lesion
   segmentation images/masks. Не подменяйте набор отдельным mirror или
   challenge-only unlabeled test archive.
3. Распакуйте, сохранив official train/test filenames, под
   `$FUZZYXAI_CH6_DATA_ROOT/eyes/idrid/` в структуру ниже.
4. Выполните сначала проверку, затем preparation:

```bash
PYTHONPATH=. python chapter6_medical_validation/ophthalmology/scripts/prepare_datasets.py --verify-only
PYTHONPATH=. python chapter6_medical_validation/ophthalmology/scripts/prepare_datasets.py
```

После законного получения разместите данные вне репозитория:

```text
$FUZZYXAI_CH6_DATA_ROOT/eyes/idrid/
  grading/train_labels.csv
  grading/test_labels.csv
  grading/train_images/
  grading/test_images/
  segmentation/train_images/
  segmentation/test_images/
  segmentation/train_masks/
  segmentation/test_masks/
```

Ожидаются официальные 413 grading train и 103 grading test изображений. Train
делится stratified с seed 2026; test не используется до freeze модели,
preprocessing и calibration. Реальные counts/hashes должны быть вычислены
loader-ом, а не перенесены в итоговые результаты из документации.

До появления данных нельзя честно сформировать checkpoints, medical metrics,
Grad-CAM/IG lesion diagnostics или FuzzyXAI ophthalmology cases. Этот blocker
не препятствует независимым ECG и Allen brain экспериментам.
