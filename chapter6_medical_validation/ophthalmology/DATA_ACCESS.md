# Official data access and placement

IDRiD was not found in the implementation environment on 2026-08-28.
The official route reaches IEEE DataPort but requires interactive account
access/acceptance. Download is intentionally not attempted through an
unofficial mirror. See `DATA_ACCESS_REQUIRED_IDRID.md` for the exact blocker.

## APTOS2019 (non-primary retained scaffold)

1. Sign in to Kaggle.
2. Open the official competition data page and accept its rules:
   https://www.kaggle.com/competitions/aptos2019-blindness-detection/data
3. Download `train.csv` and `train_images/` only for this experiment.
4. Place them under `$FUZZYXAI_CH6_DATA_ROOT/aptos2019/`.

Expected:

```text
aptos2019/
  train.csv
  train_images/
    <id_code>.png
```

The public unlabeled competition test set and all hidden/private labels are
outside this protocol. Kaggle credentials must remain outside the repository.

## IDRiD

1. Follow the official download link from
   https://idrid.grand-challenge.org/Data/ to IEEE DataPort.
2. Preserve the official grading train/test and segmentation train/test
   directories when extracting under `$FUZZYXAI_CH6_DATA_ROOT/eyes/idrid/`.
3. Do not copy raw images or masks into this repository.

`prepare_datasets.py --verify-only` validates registered counts, labels and
paths without writing a split. A missing path is reported explicitly and stops
the command. SHA256 inventories contain relative paths only.
