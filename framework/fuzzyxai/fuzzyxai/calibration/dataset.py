from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Mapping, Sequence
import numpy as np

BETA_KEYS = ['repr', 'rules', 'activations', 'uncertainty', 'trace', 'reduction']


@dataclass(frozen=True)
class CalibrationPair:
    features: Dict[str, float]
    expert_score: float


def synthetic_calibration_pairs(n: int = 60, seed: int = 42) -> list[CalibrationPair]:
    """Generate reproducible expert-like disagreement labels for proof reports."""
    rng = np.random.default_rng(seed)
    true_w = np.asarray([0.30, 0.23, 0.13, 0.18, 0.08, 0.08])
    rows = []
    for _ in range(n):
        x = rng.beta(1.6, 3.0, size=len(BETA_KEYS))
        noise = rng.normal(0, 0.025)
        y = float(np.clip(x @ true_w + noise, 0.0, 1.0))
        rows.append(CalibrationPair({k: float(v) for k, v in zip(BETA_KEYS, x)}, y))
    return rows
