"""Within-stratum conditional rule ablation without global permutation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Sequence

import numpy as np

from .nonrefit import RuleEffectEstimate


@dataclass(frozen=True)
class ConditionalRule:
    rule_id: str
    feature_indices: tuple[int, ...]
    support: float
    redundancy: float
    bootstrap_stability: float
    predicted_class: int
    length: int
    depth: int
    subgroup_size: int


def within_stratum_resample(
    values: np.ndarray,
    feature_indices: Sequence[int],
    strata: Sequence[object],
    *,
    seed: int = 4201,
) -> np.ndarray:
    if not feature_indices:
        raise ValueError("conditional ablation requires rule feature indices")
    result = np.asarray(values).copy()
    groups = np.asarray(strata)
    if len(groups) != len(result):
        raise ValueError("strata and values must align")
    rng = np.random.default_rng(seed)
    for group in np.unique(groups):
        rows = np.flatnonzero(groups == group)
        if len(rows) < 2:
            continue
        donors = rng.permutation(rows)
        if np.any(donors == rows):
            donors = np.roll(rows, 1)
        result[np.ix_(rows, tuple(feature_indices))] = values[np.ix_(donors, tuple(feature_indices))]
    return result


def conditional_effect(
    model: object,
    rule: ConditionalRule,
    values: np.ndarray,
    labels: np.ndarray,
    strata: Sequence[object],
    *,
    metric: Callable[[np.ndarray, np.ndarray], float],
    seed: int = 4201,
) -> RuleEffectEstimate:
    baseline = np.asarray(model.predict(values))  # type: ignore[attr-defined]
    resampled = within_stratum_resample(values, rule.feature_indices, strata, seed=seed)
    ablated = np.asarray(model.predict(resampled))  # type: ignore[attr-defined]
    baseline_metric = float(metric(labels, baseline))
    ablated_metric = float(metric(labels, ablated))
    return RuleEffectEstimate(
        "conditional",
        baseline_metric,
        ablated_metric,
        baseline_metric - ablated_metric,
        len(labels),
        {"resampling": "within_stratum", "seed": seed, "global_permutation": False},
    )
