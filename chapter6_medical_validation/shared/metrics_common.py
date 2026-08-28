from __future__ import annotations

from typing import Any

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    log_loss,
    precision_score,
    recall_score,
    roc_auc_score,
)


def ece_binary(labels: np.ndarray, probabilities: np.ndarray, bins: int = 15) -> float:
    truth = np.asarray(labels, dtype=int)
    prob = np.asarray(probabilities, dtype=float)
    confidence = np.maximum(prob, 1 - prob)
    predicted = (prob >= 0.5).astype(int)
    edges = np.linspace(0, 1, bins + 1)
    value = 0.0
    for index in range(bins):
        mask = (confidence > edges[index]) & (confidence <= edges[index + 1])
        if index == 0:
            mask |= confidence == edges[index]
        if mask.any():
            value += float(mask.mean()) * abs(float((predicted[mask] == truth[mask]).mean()) - float(confidence[mask].mean()))
    return value


def binary_metrics(labels: np.ndarray, probabilities: np.ndarray) -> dict[str, Any]:
    truth = np.asarray(labels, dtype=int)
    prob = np.asarray(probabilities, dtype=float)
    if truth.shape != prob.shape or not np.isfinite(prob).all() or np.any((prob < 0) | (prob > 1)):
        raise ValueError("binary metric inputs are invalid")
    predicted = (prob >= 0.5).astype(int)
    matrix = confusion_matrix(truth, predicted, labels=[0, 1])
    tn, fp, fn, tp = matrix.ravel()
    return {"n": len(truth), "accuracy": float(accuracy_score(truth, predicted)), "balanced_accuracy": float(balanced_accuracy_score(truth, predicted)), "f1": float(f1_score(truth, predicted, zero_division=0)), "precision": float(precision_score(truth, predicted, zero_division=0)), "recall_sensitivity": float(recall_score(truth, predicted, zero_division=0)), "specificity": float(tn / max(tn + fp, 1)), "auroc": float(roc_auc_score(truth, prob)), "auprc": float(average_precision_score(truth, prob)), "confusion_matrix": matrix.tolist(), "nll": float(log_loss(truth, np.column_stack((1 - prob, prob)), labels=[0, 1])), "brier": float(np.mean(np.square(prob - truth))), "ece_15_bin": ece_binary(truth, prob), "tp": int(tp), "tn": int(tn), "fp": int(fp), "fn": int(fn)}
