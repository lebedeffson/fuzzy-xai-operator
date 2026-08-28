"""Deterministic experiment-side LIME for registered PAPILA ROI inputs.

This is native-XAI evidence.  It deliberately does not calculate FuzzyXAI
system Gamma, Delta, risk, or action.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np
from sklearn.linear_model import Ridge
from skimage.segmentation import slic


@dataclass(frozen=True)
class LimeImageResult:
    superpixels: np.ndarray
    coefficients: np.ndarray
    intercept: float
    local_fit_r2: float
    target_class: int
    perturbation_hash: str
    positive_map: np.ndarray
    signed_map: np.ndarray
    negative_coefficients: dict[int, float]


def explain_lime(image: np.ndarray, predict_probabilities: Callable[[np.ndarray], np.ndarray], *, target_class: int, seed: int = 2026, n_segments: int = 50, compactness: float = 10.0, sigma: float = 1.0, n_perturbations: int = 1000, kernel_width: float = 0.25, batch_size: int = 32) -> LimeImageResult:
    """Fit a weighted local linear surrogate over fixed SLIC perturbations."""
    import hashlib

    rgb = np.asarray(image, dtype=np.float32)
    if rgb.ndim != 3 or rgb.shape[2] != 3: raise ValueError("LIME requires HxWx3 ROI")
    segments = slic(rgb, n_segments=n_segments, compactness=compactness, sigma=sigma, start_label=0, channel_axis=-1)
    feature_count = int(segments.max()) + 1; rng = np.random.default_rng(seed)
    binary = rng.integers(0, 2, size=(n_perturbations, feature_count), dtype=np.int8); binary[0] = 1
    baseline = rgb.mean(axis=(0, 1), keepdims=True); predictions: list[np.ndarray] = []
    for start in range(0, n_perturbations, batch_size):
        block = binary[start:start + batch_size]; batch = np.repeat(rgb[None, ...], len(block), axis=0)
        for index, row in enumerate(block): batch[index, ~row.astype(bool)[segments]] = baseline
        values = np.asarray(predict_probabilities(batch), dtype=float)
        if values.ndim != 2 or values.shape[1] <= target_class: raise ValueError("LIME prediction function returned incompatible probabilities")
        predictions.append(values[:, target_class])
    response = np.concatenate(predictions); distance = np.sqrt(np.mean((binary - 1) ** 2, axis=1)); weights = np.exp(-(distance ** 2) / max(kernel_width ** 2, 1e-12))
    surrogate = Ridge(alpha=1.0, fit_intercept=True).fit(binary, response, sample_weight=weights); coefficients = np.asarray(surrogate.coef_, dtype=float)
    signed = coefficients[segments]; positive = np.maximum(signed, 0.0); total = float(positive.sum()); positive = positive / total if total else positive
    digest = hashlib.sha256(binary.tobytes()).hexdigest()
    return LimeImageResult(segments, coefficients, float(surrogate.intercept_), float(surrogate.score(binary, response, sample_weight=weights)), int(target_class), digest, positive, signed, {int(index): float(value) for index, value in enumerate(coefficients) if value < 0})
