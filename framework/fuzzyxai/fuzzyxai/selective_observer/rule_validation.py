"""H6 formative planted-rule checks and confirmatory low-redundancy ablation."""

from __future__ import annotations

from dataclasses import asdict
from typing import Sequence

import numpy as np

from .contracts import ConfirmatoryProtocolLock, RuleAblationObservation, RuleProfile


def rule_priority(profile: RuleProfile) -> float:
    """Rank subgroup-specific, stable and non-redundant rules on development data."""
    return (
        0.25 * profile.subgroup_specificity
        + 0.20 * profile.subgroup_coverage
        + 0.15 * profile.activation_stability
        + 0.15 * (1.0 - profile.functional_redundancy)
        + 0.15 * profile.unique_prediction_fraction
        + 0.10 * profile.margin_contribution
    )


def select_candidate_rules(profiles: Sequence[RuleProfile], *, count: int) -> list[RuleProfile]:
    if count <= 0 or len(profiles) < count:
        raise ValueError("candidate count must be positive and available")
    return sorted(profiles, key=lambda profile: (-rule_priority(profile), profile.rule_id))[:count]


def matched_controls(candidate: RuleProfile, profiles: Sequence[RuleProfile], *, count: int = 5) -> list[RuleProfile]:
    alternatives = [profile for profile in profiles if profile.rule_id != candidate.rule_id]
    if len(alternatives) < count:
        raise ValueError("at least five control rules are required")
    alternatives.sort(
        key=lambda profile: (
            abs(profile.subgroup_coverage - candidate.subgroup_coverage),
            abs(profile.activation_stability - candidate.activation_stability),
            abs(profile.depth - candidate.depth),
            abs(profile.length - candidate.length),
            abs(profile.functional_redundancy - candidate.functional_redundancy),
            profile.rule_id,
        )
    )
    return alternatives[:count]


def inject_planted_rule(
    values: np.ndarray,
    labels: np.ndarray,
    *,
    feature: int,
    quantile: float = 0.85,
    target_class: int = 1,
) -> tuple[np.ndarray, np.ndarray, dict[str, object]]:
    """Create a formative semisynthetic label mechanism without touching confirmatory data."""
    if values.ndim != 2 or labels.ndim != 1 or len(values) != len(labels):
        raise ValueError("values and labels must be aligned tabular arrays")
    if not 0.5 <= quantile < 1.0 or not 0 <= feature < values.shape[1]:
        raise ValueError("invalid planted-rule definition")
    threshold = float(np.quantile(values[:, feature], quantile))
    mask = np.asarray(values[:, feature] >= threshold, dtype=bool)
    result = np.asarray(labels).copy()
    result[mask] = target_class
    return (
        result,
        mask,
        {
            "phase": "formative_semisynthetic",
            "rule": f"feature_{feature} >= {threshold:.12g}",
            "feature": feature,
            "threshold": threshold,
            "target_class": target_class,
            "affected_objects": int(mask.sum()),
            "confirmatory_claim_allowed": False,
        },
    )


def evaluate_planted_rule_recovery(
    profiles: Sequence[RuleProfile],
    planted_rule_ids: Sequence[str],
    *,
    top_k: int,
) -> dict[str, object]:
    selected = select_candidate_rules(profiles, count=top_k)
    selected_ids = {profile.rule_id for profile in selected}
    truth = set(planted_rule_ids)
    true_positive = len(selected_ids & truth)
    precision = true_positive / max(1, len(selected_ids))
    recall = true_positive / max(1, len(truth))
    return {
        "phase": "formative_semisynthetic",
        "top_k": top_k,
        "selected": [asdict(profile) for profile in selected],
        "planted_rule_precision": precision,
        "planted_rule_recall": recall,
        "methodology_supported": precision >= 0.80 and recall >= 0.80,
        "real_data_claim_allowed": False,
    }


def evaluate_confirmatory_ablation(
    observations: Sequence[RuleAblationObservation],
    protocol_lock: ConfirmatoryProtocolLock,
    *,
    minimum_specific_effect: float = 0.01,
    bootstrap_repetitions: int = 2000,
    seed: int = 4201,
) -> dict[str, object]:
    if not observations:
        raise ValueError("confirmatory ablation requires held-out observations")
    effects = np.asarray(
        [item.candidate_effect - float(np.median(item.matched_control_effects)) for item in observations],
        dtype=float,
    )
    interval = _bootstrap_interval(effects, bootstrap_repetitions, seed)
    datasets = sorted({item.dataset_id for item in observations})
    by_dataset = {
        dataset: float(np.mean([item.candidate_effect - float(np.median(item.matched_control_effects)) for item in observations if item.dataset_id == dataset]))
        for dataset in datasets
    }
    supported = len(datasets) >= 2 and interval[0] > minimum_specific_effect and all(value > 0 for value in by_dataset.values())
    return {
        "endpoint": "candidate_effect_minus_median_matched_control_effect",
        "protocol_sha256": protocol_lock.protocol_sha256,
        "n_observations": len(observations),
        "datasets": datasets,
        "mean_specific_effect": float(effects.mean()),
        "confidence_interval_95": interval,
        "minimum_specific_effect": minimum_specific_effect,
        "dataset_effects": by_dataset,
        "status": "supported" if supported else "not_supported",
        "claim_allowed": supported,
        "allowed_wording": (
            "Low-redundancy subgroup rules had a larger held-out specific effect than matched controls."
            if supported
            else "Rule ablation remains a local diagnostic; no general subgroup-rule effect is claimed."
        ),
    }


def _bootstrap_interval(values: np.ndarray, repetitions: int, seed: int) -> list[float]:
    rng = np.random.default_rng(seed)
    means = [float(np.mean(values[rng.integers(0, len(values), size=len(values))])) for _ in range(repetitions)]
    return [float(np.quantile(means, 0.025)), float(np.quantile(means, 0.975))]
