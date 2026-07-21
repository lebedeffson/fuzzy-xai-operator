"""Rule-ablation contracts that keep distinct estimands separate."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class AblationEstimate:
    estimand: str
    effect: float
    support: float
    redundancy: float
    sign_correct: bool | None


def non_refit_ablation(base_metric: float, metric_without_rule: float, *, support: float, redundancy: float) -> AblationEstimate:
    return AblationEstimate("current_model_rule_dependence", base_metric - metric_without_rule, support, redundancy, None)


def refit_ablation(base_metric: float, refit_metric_without_structure: float, *, support: float, redundancy: float) -> AblationEstimate:
    return AblationEstimate("information_replaceability", base_metric - refit_metric_without_structure, support, redundancy, None)


def conditional_permutation_effect(
    values: Sequence[float],
    strata: Sequence[str],
    scorer: Callable[[np.ndarray], float],
    *,
    seed: int,
) -> float:
    array, groups = np.asarray(values, dtype=float), np.asarray(strata)
    if len(array) != len(groups):
        raise ValueError("conditional permutation inputs must align")
    rng = np.random.default_rng(seed)
    permuted = array.copy()
    for group in np.unique(groups):
        positions = np.flatnonzero(groups == group)
        permuted[positions] = rng.permutation(permuted[positions])
    return float(scorer(array) - scorer(permuted))


def eligible_region(*, effect: float, support: float, redundancy: float) -> bool:
    return effect >= 0.05 and support >= 0.05 and redundancy <= 0.50
