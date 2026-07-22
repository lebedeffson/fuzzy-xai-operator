"""Ablation of equivalence clusters and minimal sufficient rule subsets."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Sequence

import numpy as np

from .equivalence import RuleEquivalenceCluster


@dataclass(frozen=True)
class ClusterAblationEffect:
    cluster_id: str
    rule_ids: tuple[str, ...]
    individual_effects: tuple[float, ...]
    cluster_effect: float
    synergy: float
    baseline_metric: float
    ablated_metric: float


def assess_cluster_ablation(
    model: object,
    cluster: RuleEquivalenceCluster,
    values: np.ndarray,
    labels: np.ndarray,
    *,
    metric: Callable[[np.ndarray, np.ndarray], float],
) -> ClusterAblationEffect:
    if not hasattr(model, "predict_without_rule") or not hasattr(model, "predict_without_rules"):
        raise TypeError("cluster ablation requires predict_without_rule and predict_without_rules")
    baseline = float(metric(labels, np.asarray(model.predict(values))))  # type: ignore[attr-defined]
    individual = tuple(
        baseline - float(metric(labels, np.asarray(model.predict_without_rule(values, rule_id))))  # type: ignore[attr-defined]
        for rule_id in cluster.rule_ids
    )
    cluster_metric = float(metric(labels, np.asarray(model.predict_without_rules(values, cluster.rule_ids))))  # type: ignore[attr-defined]
    cluster_effect = baseline - cluster_metric
    return ClusterAblationEffect(cluster.cluster_id, cluster.rule_ids, individual, cluster_effect, cluster_effect - max(individual, default=0.0), baseline, cluster_metric)


def minimal_sufficient_subset(
    rule_ids: Sequence[str],
    score_without: Callable[[tuple[str, ...]], float],
    *,
    maximum_loss: float,
) -> tuple[str, ...]:
    selected = list(rule_ids)
    changed = True
    while changed:
        changed = False
        for rule_id in tuple(selected):
            candidate = tuple(item for item in selected if item != rule_id)
            if score_without(candidate) <= maximum_loss:
                selected.remove(rule_id)
                changed = True
    return tuple(selected)
