from __future__ import annotations

from collections.abc import Iterable

import numpy as np


def fit_lead_statistics(signals: Iterable[np.ndarray]) -> dict[str, list[float] | str]:
    count = 0
    total = np.zeros(12, dtype=np.float64)
    total_sq = np.zeros(12, dtype=np.float64)
    for signal in signals:
        value = np.asarray(signal, dtype=np.float64)
        if value.shape != (12, 1000) or not np.isfinite(value).all():
            raise ValueError("train statistic input must be finite 12x1000")
        count += value.shape[1]
        total += value.sum(axis=1)
        total_sq += np.square(value).sum(axis=1)
    if count == 0:
        raise ValueError("cannot fit normalization without train signals")
    mean = total / count
    variance = np.maximum(total_sq / count - np.square(mean), 1e-12)
    return {"status": "fitted_on_train_only", "lead_mean": mean.tolist(), "lead_std": np.sqrt(variance).tolist(), "samples_per_lead": count}


def normalize(signal: np.ndarray, statistics: dict[str, object]) -> np.ndarray:
    value = np.asarray(signal, dtype=np.float32)
    mean = np.asarray(statistics["lead_mean"], dtype=np.float32)[:, None]
    std = np.asarray(statistics["lead_std"], dtype=np.float32)[:, None]
    normalized = (value - mean) / std
    if normalized.shape != (12, 1000) or not np.isfinite(normalized).all():
        raise ValueError("ECG normalization produced invalid output")
    return normalized


def technical_signal_quality(signal: np.ndarray) -> dict[str, object]:
    value = np.asarray(signal, dtype=float)
    variance = value.var(axis=1)
    return {"semantics": "technical_signal_quality_evidence_not_diagnosis", "finite_fraction": float(np.isfinite(value).mean()), "flatline_fraction_per_lead": (np.abs(np.diff(value, axis=1)) < 1e-8).mean(axis=1).tolist(), "extreme_fraction_per_lead": (np.abs(value) > 20).mean(axis=1).tolist(), "variance_per_lead": variance.tolist()}
