"""Matched-budget policy baselines for formative and locked confirmation."""

from __future__ import annotations

from collections.abc import Mapping, Sequence

import numpy as np

from fuzzyxai.selective_observer import SelectiveAction


BASELINE_POLICIES = (
    "always_accept",
    "always_review",
    "raw_confidence_threshold",
    "calibrated_confidence_threshold",
    "uncertainty_threshold",
    "model_disagreement",
    "explainer_disagreement",
    "provenance_completeness",
    "data_quality_guardrail",
    "simple_or_guardrail",
    "weighted_linear_score",
    "predictive_risk_without_route",
    "conformal_selective",
    "full_fuzzyxai_practical_controller",
)


def allocate_score_budget(
    scores: Sequence[float],
    *,
    review_budget: float,
    hard_blocks: Sequence[bool] | None = None,
) -> list[SelectiveAction]:
    values = np.asarray(scores, dtype=float)
    if values.ndim != 1 or not len(values) or np.any(~np.isfinite(values)):
        raise ValueError("policy scores must be a finite vector")
    if not 0.0 <= review_budget <= 1.0:
        raise ValueError("review budget must be in [0, 1]")
    blocks = np.zeros(len(values), dtype=bool) if hard_blocks is None else np.asarray(hard_blocks, dtype=bool)
    if len(blocks) != len(values):
        raise ValueError("hard blocks must align with policy scores")
    # NumPy coerces ``str`` enums during assignment and may silently truncate
    # them to values such as ``"Select"``. Keep actions as Python objects so
    # policy accounting cannot lose the enum identity.
    actions = [SelectiveAction.ACCEPT for _ in values]
    for index in np.flatnonzero(blocks):
        actions[int(index)] = SelectiveAction.BLOCK
    eligible = np.flatnonzero(~blocks)
    count = min(len(eligible), int(review_budget * len(values) + 1e-12))
    order = eligible[np.argsort(-values[eligible], kind="stable")]
    for index in order[:count]:
        actions[int(index)] = SelectiveAction.FULL_REVIEW
    return actions


def policy_metrics(
    invalid_outcomes: Sequence[bool],
    actions: Sequence[SelectiveAction],
    *,
    true_hard_faults: Sequence[bool] | None = None,
) -> dict[str, float | int]:
    invalid = np.asarray(invalid_outcomes, dtype=bool)
    if len(invalid) != len(actions) or not len(invalid):
        raise ValueError("policy outcomes and actions must be aligned")
    accepted = np.asarray([action is SelectiveAction.ACCEPT for action in actions])
    blocked = np.asarray([action is SelectiveAction.BLOCK for action in actions])
    reviewed = np.asarray([action in {SelectiveAction.SHORT_REVIEW, SelectiveAction.FULL_REVIEW} for action in actions])
    hard = np.zeros(len(invalid), dtype=bool) if true_hard_faults is None else np.asarray(true_hard_faults, dtype=bool)
    wrong = invalid & accepted
    false_blocks = blocked & ~hard
    if not np.all(accepted | reviewed | blocked):
        raise ValueError("policy actions contain an unsupported action value")
    return {
        "n": len(invalid),
        "automatic_coverage": float(accepted.mean()),
        "review_rate": float(reviewed.mean()),
        "block_rate": float(blocked.mean()),
        "wrong_or_invalid_automatic_actions": int(wrong.sum()),
        "operational_risk": float(wrong.sum() / max(1, accepted.sum())),
        "false_blocks": int(false_blocks.sum()),
        "false_block_rate": float(false_blocks.mean()),
        "unsafe_accept_indicator_count": int(wrong.sum()),
    }


def compare_at_matched_budgets(
    invalid_outcomes: Sequence[bool],
    policy_scores: Mapping[str, Sequence[float]],
    *,
    budgets: Sequence[float],
    hard_blocks: Sequence[bool],
) -> list[dict[str, object]]:
    missing = set(BASELINE_POLICIES) - set(policy_scores)
    if missing:
        raise ValueError(f"missing preregistered policy scores: {sorted(missing)}")
    result = []
    for budget in budgets:
        for policy_name in BASELINE_POLICIES:
            if policy_name == "always_accept":
                actions = [SelectiveAction.ACCEPT] * len(invalid_outcomes)
            elif policy_name == "always_review":
                actions = [SelectiveAction.FULL_REVIEW] * len(invalid_outcomes)
            else:
                actions = allocate_score_budget(
                    policy_scores[policy_name],
                    review_budget=float(budget),
                    hard_blocks=hard_blocks if policy_name == "full_fuzzyxai_practical_controller" else None,
                )
            result.append({"policy": policy_name, "review_budget": float(budget), **policy_metrics(invalid_outcomes, actions, true_hard_faults=hard_blocks)})
    return result


def component_ablation_scores(
    predictive_score: Sequence[float],
    components: Mapping[str, Sequence[float]],
) -> dict[str, list[float]]:
    predictive = np.asarray(predictive_score, dtype=float)
    arrays = {name: np.asarray(values, dtype=float) for name, values in components.items()}
    if any(len(values) != len(predictive) for values in arrays.values()):
        raise ValueError("ablation components must be object-aligned")
    full_route = np.maximum.reduce(list(arrays.values())) if arrays else np.zeros(len(predictive))
    result = {"P0_predictive_only": predictive.tolist(), "P1_full": (1.0 - (1.0 - predictive) * (1.0 - full_route)).tolist()}
    for removed in arrays:
        retained = [values for name, values in arrays.items() if name != removed]
        route = np.maximum.reduce(retained) if retained else np.zeros(len(predictive))
        result[f"P1_minus_{removed}"] = (1.0 - (1.0 - predictive) * (1.0 - route)).tolist()
    return result
