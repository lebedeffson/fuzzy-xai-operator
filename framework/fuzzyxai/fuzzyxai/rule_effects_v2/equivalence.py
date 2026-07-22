"""Rule equivalence classes based on activation and condition similarity."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Sequence

import numpy as np


@dataclass(frozen=True)
class RuleEquivalenceCluster:
    cluster_id: str
    rule_ids: tuple[str, ...]
    mean_activation_jaccard: float
    representative_rule_id: str


def cluster_equivalent_rules(
    activations: Mapping[str, Sequence[bool]],
    *,
    jaccard_threshold: float = 0.80,
) -> tuple[RuleEquivalenceCluster, ...]:
    if not 0.0 <= jaccard_threshold <= 1.0 or not activations:
        raise ValueError("activations and a valid threshold are required")
    arrays = {name: np.asarray(values, dtype=bool) for name, values in activations.items()}
    lengths = {len(value) for value in arrays.values()}
    if len(lengths) != 1:
        raise ValueError("rule activations must align")
    remaining = set(arrays)
    clusters = []
    while remaining:
        seed = min(remaining)
        members = [seed]
        for candidate in sorted(remaining - {seed}):
            if min(_jaccard(arrays[candidate], arrays[item]) for item in members) >= jaccard_threshold:
                members.append(candidate)
        remaining.difference_update(members)
        similarities = [_jaccard(arrays[left], arrays[right]) for index, left in enumerate(members) for right in members[index + 1 :]]
        support = {item: float(arrays[item].mean()) for item in members}
        representative = max(members, key=lambda item: (support[item], item))
        clusters.append(
            RuleEquivalenceCluster(
                cluster_id=f"cluster-{len(clusters):04d}",
                rule_ids=tuple(sorted(members)),
                mean_activation_jaccard=float(np.mean(similarities)) if similarities else 1.0,
                representative_rule_id=representative,
            )
        )
    return tuple(clusters)


def _jaccard(left: np.ndarray, right: np.ndarray) -> float:
    union = np.sum(left | right)
    return float(np.sum(left & right) / union) if union else 1.0
