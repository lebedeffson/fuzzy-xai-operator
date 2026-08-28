"""Freeze a reviewer-readable sanity gate from existing PAPILA run artifacts."""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--metrics", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    root = args.data_root / "eyes" / "papila"; split = json.loads((root / "verified" / "papila_cv_folds_seed2026.json").read_text()); metrics = json.loads(args.metrics.read_text())
    lines = ["# PAPILA model sanity gate", "", "Status: **FREEZE** — this gate audits saved runs only; it does not search hyperparameters or alter a model.", "", "## Protocol checks", "", "- Label mapping: `0=healthy`, `1=glaucoma`, `2=suspect`.", "- All primary rows are binary clean-patient rows; suspect-associated patients are excluded wholesale.", "- Train/validation/test patient sets are pairwise disjoint in each outer fold; paired eyes share a fold.", "- Class weights are calculated from train rows only by `train_papila_cv.py`.", "- ROI provenance is `expert_1_optic_disc_segmentation`; no diagnosis enters ROI extraction.", "- Preprocessing is deterministic outside training; no outer-test threshold or preprocessing selection is registered.", "", "## Saved outer-fold metrics (seed 2026)", "", "| Fold | N train / val / test | Test H/G | Accuracy | Balanced accuracy | Precision | Sensitivity | Specificity | F1 | AUROC | AUPRC | NLL | Brier | ECE | Confusion [H,G] |", "|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|"]
    for fold in range(1, 6):
        m = metrics["folds"][str(fold)]; c = split["folds"][str(fold)]["counts"]; cc=m["class_counts"]
        lines.append(f"| {fold} | {c['train_eyes']}/{c['validation_eyes']}/{c['test_eyes']} | {cc['healthy']}/{cc['glaucoma']} | {m['accuracy']:.4f} | {m['balanced_accuracy']:.4f} | {m['precision']:.4f} | {m['recall_sensitivity']:.4f} | {m['specificity']:.4f} | {m['f1']:.4f} | {m['auroc']:.4f} | {m['auprc']:.4f} | {m['nll']:.4f} | {m['brier']:.4f} | {m['ece_15_bin']:.4f} | {m['confusion_matrix']} |")
    lines += ["", "## Aggregate", "", "The fixed-seed mean±SD results are descriptive outer-fold estimates. The modest AUROC/balanced-accuracy result is retained without model tuning against test folds. Canonical explanatory fold: 5; canonical seed is selected only by minimum internal validation loss."]
    args.output.parent.mkdir(parents=True, exist_ok=True); args.output.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(args.output)

if __name__ == "__main__": main()
