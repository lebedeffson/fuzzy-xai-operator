from __future__ import annotations

from collections.abc import Callable
from typing import Any, cast

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    cohen_kappa_score,
    confusion_matrix,
    f1_score,
    log_loss,
    recall_score,
    roc_auc_score,
)


def validate_probabilities(probabilities: np.ndarray, *, classes: int = 5) -> np.ndarray:
    values = np.asarray(probabilities, dtype=float)
    if values.ndim != 2 or values.shape[1] != classes:
        raise ValueError(f"probability matrix must have shape N x {classes}")
    if not np.isfinite(values).all() or np.any(values < 0):
        raise ValueError("probabilities must be finite and non-negative")
    if not np.allclose(values.sum(axis=1), 1.0, atol=1e-6):
        raise ValueError("probability rows must sum to one")
    return cast(np.ndarray, values)


def expected_calibration_error(y_true: np.ndarray, probabilities: np.ndarray, *, bins: int = 15) -> float:
    values = validate_probabilities(probabilities)
    truth = np.asarray(y_true, dtype=int)
    predicted = values.argmax(axis=1)
    confidence = values.max(axis=1)
    edges = np.linspace(0.0, 1.0, bins + 1)
    total = len(truth)
    score = 0.0
    for index in range(bins):
        low, high = edges[index], edges[index + 1]
        mask = (confidence > low) & (confidence <= high) if index else (confidence >= low) & (confidence <= high)
        if not mask.any():
            continue
        score += float(mask.mean()) * abs(float((predicted[mask] == truth[mask]).mean()) - float(confidence[mask].mean()))
    return score if total else float("nan")


def multiclass_brier(y_true: np.ndarray, probabilities: np.ndarray) -> float:
    values = validate_probabilities(probabilities)
    one_hot = np.eye(values.shape[1])[np.asarray(y_true, dtype=int)]
    return float(np.mean(np.sum((values - one_hot) ** 2, axis=1)))


def classification_metrics(y_true: np.ndarray, probabilities: np.ndarray) -> dict[str, Any]:
    values = validate_probabilities(probabilities)
    truth = np.asarray(y_true, dtype=int)
    predicted = values.argmax(axis=1)
    labels = list(range(values.shape[1]))
    result: dict[str, Any] = {
        "n": len(truth),
        "accuracy": float(accuracy_score(truth, predicted)),
        "macro_f1": float(f1_score(truth, predicted, labels=labels, average="macro", zero_division=0)),
        "weighted_f1": float(f1_score(truth, predicted, labels=labels, average="weighted", zero_division=0)),
        "balanced_accuracy": float(balanced_accuracy_score(truth, predicted)),
        "quadratic_weighted_kappa": float(cohen_kappa_score(truth, predicted, weights="quadratic")),
        "per_class_recall": recall_score(truth, predicted, labels=labels, average=None, zero_division=0).tolist(),
        "confusion_matrix": confusion_matrix(truth, predicted, labels=labels).tolist(),
        "multiclass_nll": float(log_loss(truth, values, labels=labels)),
        "brier_score": multiclass_brier(truth, values),
        "ece_15_bin": expected_calibration_error(truth, values, bins=15),
    }
    try:
        result["ovr_roc_auc_macro"] = float(roc_auc_score(truth, values, labels=labels, multi_class="ovr", average="macro"))
        result["ovr_roc_auc_status"] = "measured"
    except ValueError as exc:
        result["ovr_roc_auc_macro"] = None
        result["ovr_roc_auc_status"] = f"not_evaluated: {exc}"
    return result


def attribution_spatial_metrics(attribution: np.ndarray, lesion_mask: np.ndarray, *, top_fractions: tuple[float, ...] = (0.10, 0.20)) -> dict[str, Any]:
    values = np.abs(np.asarray(attribution, dtype=float))
    if values.ndim == 3:
        values = values.sum(axis=0) if values.shape[0] <= 4 else values.sum(axis=-1)
    mask = np.asarray(lesion_mask, dtype=bool)
    if values.shape != mask.shape:
        raise ValueError("attribution and lesion mask must share spatial shape")
    total = float(values.sum())
    energy = None if total <= 0 else float(values[mask].sum() / total)
    max_point = np.unravel_index(int(np.argmax(values)), values.shape)
    overlaps: dict[str, float | None] = {}
    flat = values.ravel()
    for fraction in top_fractions:
        count = max(1, int(np.ceil(flat.size * fraction)))
        threshold = np.partition(flat, -count)[-count]
        selected = values >= threshold
        union = np.logical_or(selected, mask).sum()
        overlaps[f"top_{int(fraction * 100)}_iou"] = None if union == 0 else float(np.logical_and(selected, mask).sum() / union)
    return {
        "semantics": "spatial_correspondence_not_causality_or_Gamma",
        "attribution_energy_inside_lesion_union": energy,
        "pointing_game_inside": bool(mask[max_point]),
        "max_point": [int(max_point[0]), int(max_point[1])],
        **overlaps,
    }


def deletion_faithfulness(
    image: np.ndarray,
    attribution: np.ndarray,
    predict_probabilities: Callable[[np.ndarray], np.ndarray],
    *,
    target: int,
    fraction: float = 0.10,
    random_repeats: int = 20,
    seed: int = 2026,
    baseline: float = 0.0,
) -> dict[str, Any]:
    source = np.asarray(image, dtype=float)
    heat = np.abs(np.asarray(attribution, dtype=float))
    if heat.ndim == 3:
        heat = heat.sum(axis=0) if heat.shape[0] <= 4 else heat.sum(axis=-1)
    if heat.shape != source.shape[:2]:
        raise ValueError("attribution heatmap does not match image")
    count = max(1, int(np.ceil(heat.size * fraction)))
    top_indexes = np.argpartition(heat.ravel(), -count)[-count:]

    def masked(indexes: np.ndarray) -> np.ndarray:
        value = source.copy().reshape(-1, *source.shape[2:])
        value[indexes] = baseline
        return cast(np.ndarray, value.reshape(source.shape))

    original = float(validate_probabilities(np.asarray(predict_probabilities(source[None, ...])))[0, target])
    top_probability = float(validate_probabilities(np.asarray(predict_probabilities(masked(top_indexes)[None, ...])))[0, target])
    rng = np.random.default_rng(seed)
    random_probabilities = []
    for _ in range(random_repeats):
        indexes = rng.choice(heat.size, size=count, replace=False)
        random_probabilities.append(float(validate_probabilities(np.asarray(predict_probabilities(masked(indexes)[None, ...])))[0, target]))
    top_drop = original - top_probability
    random_drop = original - float(np.mean(random_probabilities))
    return {
        "semantics": "controlled_deletion_diagnostic_not_lesion_alignment",
        "target": int(target),
        "fraction": fraction,
        "random_repeats": random_repeats,
        "seed": seed,
        "baseline": baseline,
        "original_probability": original,
        "top_masked_probability": top_probability,
        "drop_target_probability_top": top_drop,
        "drop_target_probability_random_mean": random_drop,
        "difference": top_drop - random_drop,
    }
