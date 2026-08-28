# PAPILA model sanity gate

Status: **FREEZE** — this gate audits saved runs only; it does not search hyperparameters or alter a model.

## Protocol checks

- Label mapping: `0=healthy`, `1=glaucoma`, `2=suspect`.
- All primary rows are binary clean-patient rows; suspect-associated patients are excluded wholesale.
- Train/validation/test patient sets are pairwise disjoint in each outer fold; paired eyes share a fold.
- Class weights are calculated from train rows only by `train_papila_cv.py`.
- ROI provenance is `expert_1_optic_disc_segmentation`; no diagnosis enters ROI extraction.
- Preprocessing is deterministic outside training; no outer-test threshold or preprocessing selection is registered.

## Saved outer-fold metrics (seed 2026)

| Fold | N train / val / test | Test H/G | Accuracy | Balanced accuracy | Precision | Sensitivity | Specificity | F1 | AUROC | AUPRC | NLL | Brier | ECE | Confusion [H,G] |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| 1 | 284/52/84 | 69/15 | 0.7262 | 0.5986 | 0.3000 | 0.4000 | 0.7971 | 0.3429 | 0.6696 | 0.3195 | 0.6110 | 0.1901 | 0.1200 | [[55, 14], [9, 6]] |
| 2 | 284/52/84 | 65/19 | 0.7976 | 0.6457 | 0.5833 | 0.3684 | 0.9231 | 0.4516 | 0.7142 | 0.4657 | 0.5162 | 0.1592 | 0.1418 | [[60, 5], [12, 7]] |
| 3 | 284/52/84 | 65/19 | 0.7500 | 0.4846 | 0.0000 | 0.0000 | 0.9692 | 0.0000 | 0.5838 | 0.2720 | 0.5658 | 0.1845 | 0.1221 | [[63, 2], [19, 0]] |
| 4 | 284/52/84 | 69/15 | 0.7619 | 0.5942 | 0.3333 | 0.3333 | 0.8551 | 0.3333 | 0.7343 | 0.3859 | 0.4830 | 0.1561 | 0.1079 | [[59, 10], [10, 5]] |
| 5 | 284/52/84 | 65/19 | 0.8095 | 0.6721 | 0.6154 | 0.4211 | 0.9231 | 0.5000 | 0.7352 | 0.5338 | 0.5158 | 0.1498 | 0.1028 | [[60, 5], [11, 8]] |

## Aggregate

The fixed-seed mean±SD results are descriptive outer-fold estimates. The modest AUROC/balanced-accuracy result is retained without model tuning against test folds. Canonical explanatory fold: 5; canonical seed is selected only by minimum internal validation loss.
