from __future__ import annotations

from typing import Any

import numpy as np


def temporal_occlusion(model: Any, standardized: Any, *, target: int, temperature: float = 1.0, window: int = 50, stride: int = 50, batch_size: int = 64) -> dict[str, object]:
    import torch

    if tuple(standardized.shape) != (1, 12, 1000):
        raise ValueError("temporal occlusion expects 1x12x1000")
    model.eval()
    with torch.no_grad():
        original = float(torch.softmax(model(standardized) / temperature, dim=1)[0, target].item())
    variants, locations = [], []
    for lead in range(12):
        for start in range(0, 1000 - window + 1, stride):
            value = standardized.detach().clone()
            value[0, lead, start : start + window] = 0.0
            variants.append(value)
            locations.append((lead, start))
    scores = []
    with torch.no_grad():
        for offset in range(0, len(variants), batch_size):
            batch = torch.cat(variants[offset : offset + batch_size], dim=0)
            scores.extend(torch.softmax(model(batch) / temperature, dim=1)[:, target].cpu().tolist())
    windows = 1 + (1000 - window) // stride
    importance = np.zeros((12, windows), dtype=np.float64)
    for (lead, start), probability in zip(locations, scores, strict=True):
        importance[lead, start // stride] = original - float(probability)
    return {"target": target, "baseline": "zero_standardized_train_mean", "window_samples": window, "stride_samples": stride, "original_probability": original, "probability_semantics": "temperature_scaled_probability", "temperature": float(temperature), "importance": importance}


def common_ig_representation(integrated_gradients: np.ndarray, bins: int = 20) -> np.ndarray:
    value = np.asarray(integrated_gradients, dtype=float).reshape(12, 1000)
    result = value.reshape(12, bins, 1000 // bins).sum(axis=2)
    scale = float(np.abs(result).sum())
    return result if scale == 0 else result / scale


def common_occlusion_representation(importance: np.ndarray, bins: int = 20) -> np.ndarray:
    value = np.asarray(importance, dtype=float)
    if value.shape != (12, bins):
        raise ValueError(f"registered occlusion grid must be 12x{bins}, got {value.shape}")
    scale = float(np.abs(value).sum())
    return value if scale == 0 else value / scale
