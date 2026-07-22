from __future__ import annotations

import math
from typing import Any

import numpy as np

from .models import RouteObservation
from .taxonomy import FAULT_SPECS


FEATURE_NAMES = tuple(sorted({field for spec in FAULT_SPECS for field in spec.fields}))


def field_distance(field: str, expected: Any, observed: Any) -> float:
    if observed in (None, "", (), []):
        return 1.5
    if field in {"calibration_age_days", "reduction_loss"}:
        denominator = max(abs(float(expected or 1.0)), 1e-9)
        return min(2.0, abs(float(observed) - float(expected or 0.0)) / denominator)
    if isinstance(expected, (int, float)) and isinstance(observed, (int, float)):
        return min(2.0, abs(float(observed) - float(expected)) / max(abs(float(expected)), 1.0))
    if isinstance(expected, str) and isinstance(observed, str):
        if expected == observed:
            return 0.0
        prefix = 0
        for left, right in zip(expected, observed):
            if left != right:
                break
            prefix += 1
        return 0.25 + 0.75 * (1.0 - prefix / max(len(expected), len(observed), 1))
    return 0.0 if expected == observed else 1.0


def extract_feature_dict(route: RouteObservation) -> dict[str, float]:
    values: dict[str, float] = {}
    for field in FEATURE_NAMES:
        values[field] = field_distance(field, route.expected.get(field), route.observed.get(field))
    values["missing_fraction"] = sum(route.observed.get(field) in (None, "", (), []) for field in route.mandatory_fields) / max(
        len(route.mandatory_fields), 1
    )
    values["path_disagreement"] = sum(any(values.get(node, 0.0) > 0.0 for node in path) for path in route.dependency_paths) / max(
        len(route.dependency_paths), 1
    )
    return values


def vectorize(features: dict[str, float], names: tuple[str, ...]) -> np.ndarray:
    return np.asarray([float(features.get(name, 0.0)) for name in names], dtype=float)


def softmax(values: np.ndarray) -> np.ndarray:
    shifted = values - np.max(values)
    exp = np.exp(np.clip(shifted, -50.0, 50.0))
    return exp / max(float(np.sum(exp)), math.ulp(1.0))
