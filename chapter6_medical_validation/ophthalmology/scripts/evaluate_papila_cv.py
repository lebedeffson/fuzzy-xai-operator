"""Aggregate only saved PAPILA outer-test predictions; no model selection here."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.metrics import accuracy_score, average_precision_score, balanced_accuracy_score, brier_score_loss, confusion_matrix, f1_score, log_loss, precision_score, recall_score, roc_auc_score


def _metrics(rows: list[dict[str, Any]]) -> dict[str, Any]:
    truth = np.asarray([row["label"] for row in rows], dtype=int)
    prob = np.asarray([row["probabilities"] for row in rows], dtype=float)
    # JSON float round-trip can move a softmax row a few ulps away from one;
    # this is presentation normalization, not recalibration or model fitting.
    prob = prob / prob.sum(axis=1, keepdims=True)
    predicted = prob.argmax(axis=1)
    confidence = prob.max(axis=1); ece = 0.0
    for lower, upper in zip(np.linspace(0.0, 1.0, 16)[:-1], np.linspace(0.0, 1.0, 16)[1:], strict=True):
        mask = (confidence >= lower) & (confidence <= upper) if lower == 0 else (confidence > lower) & (confidence <= upper)
        if mask.any(): ece += float(mask.mean()) * abs(float((predicted[mask] == truth[mask]).mean()) - float(confidence[mask].mean()))
    specificity = recall_score(truth, predicted, pos_label=0, zero_division=0)
    return {"n": len(rows), "class_counts": {"healthy": int((truth == 0).sum()), "glaucoma": int((truth == 1).sum())}, "accuracy": float(accuracy_score(truth, predicted)), "balanced_accuracy": float(balanced_accuracy_score(truth, predicted)), "f1": float(f1_score(truth, predicted, pos_label=1, zero_division=0)), "precision": float(precision_score(truth, predicted, pos_label=1, zero_division=0)), "recall_sensitivity": float(recall_score(truth, predicted, pos_label=1, zero_division=0)), "specificity": float(specificity), "auroc": float(roc_auc_score(truth, prob[:, 1])), "auprc": float(average_precision_score(truth, prob[:, 1])), "nll": float(log_loss(truth, prob, labels=[0, 1])), "brier": float(brier_score_loss(truth, prob[:, 1])), "ece_15_bin": ece, "confusion_matrix": confusion_matrix(truth, predicted, labels=[0, 1]).tolist()}


def main() -> None:
    parser = argparse.ArgumentParser(description="Aggregate PAPILA fixed-seed outer-fold metrics")
    parser.add_argument("--runs", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(); fold_rows: dict[str, Any] = {}; all_rows: list[dict[str, Any]] = []
    for fold in range(1, 6):
        choices = sorted(args.runs.glob(f"papila-resnet50-fold{fold}-seed{args.seed}-*/test_predictions.json"))
        if len(choices) != 1: raise FileNotFoundError(f"expected exactly one fold-{fold} seed-{args.seed} test output; found {choices}")
        rows = json.loads(choices[0].read_text(encoding="utf-8"))["rows"]; fold_rows[str(fold)] = _metrics(rows); all_rows.extend(rows)
    keys = [key for key in fold_rows["1"] if key not in {"n", "class_counts", "confusion_matrix"}]
    summary = {key: {"mean": float(np.mean([fold_rows[str(fold)][key] for fold in range(1, 6)])), "sd": float(np.std([fold_rows[str(fold)][key] for fold in range(1, 6)], ddof=1))} for key in keys}
    payload = {"schema_version": "1.0", "selection": "fixed seed 2026; every outer fold; no best-test selection", "folds": fold_rows, "pooled_outer_test": _metrics(all_rows), "mean_sd_over_folds": summary}
    args.output.parent.mkdir(parents=True, exist_ok=True); args.output.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"); print(args.output)


if __name__ == "__main__":
    main()
