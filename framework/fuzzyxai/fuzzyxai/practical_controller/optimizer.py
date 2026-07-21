"""Budget-constrained action allocation with explicit infeasibility reporting."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

from fuzzyxai.selective_observer import SelectiveAction

from .contracts import CostProfile, GuardResult, HardGuardStatus, PracticalPolicy


@dataclass(frozen=True)
class ActionCandidate:
    index: int
    risk: float
    guard: GuardResult


def allocate_actions(
    candidates: Sequence[ActionCandidate],
    *,
    review_budget: float,
    cost_profile: CostProfile,
    policy: PracticalPolicy,
) -> tuple[list[SelectiveAction], list[tuple[str, ...]], bool]:
    if not candidates or not 0.0 <= review_budget <= 1.0:
        raise ValueError("action allocation requires candidates and a valid budget")
    actions = [SelectiveAction.ACCEPT] * len(candidates)
    reasons: list[list[str]] = [[] for _ in candidates]
    mandatory = [item for item in candidates if item.guard.status is HardGuardStatus.REVIEW_REQUIRED]
    blocked = [item for item in candidates if item.guard.status is HardGuardStatus.BLOCKED]
    for item in blocked:
        actions[item.index] = SelectiveAction.BLOCK
        reasons[item.index].append("HARD_GUARD_BLOCK")

    review_capacity = int(review_budget * len(candidates) + 1e-12)
    budget_feasible = len(mandatory) <= review_capacity
    selected: set[int] = set()
    for item in sorted(mandatory, key=lambda row: (-row.risk, row.index)):
        actions[item.index] = SelectiveAction.FULL_REVIEW
        reasons[item.index].append("MANDATORY_ROUTE_REVIEW")
        selected.add(item.index)

    remaining_capacity = max(0, review_capacity - len(selected))
    optional = [item for item in candidates if item.index not in selected and item.guard.status is HardGuardStatus.CERTIFIED]
    ranked = sorted(optional, key=lambda row: (-_review_value(row.risk, cost_profile), row.index))
    for item in ranked[:remaining_capacity]:
        if item.risk > policy.full_review_max_risk:
            actions[item.index] = SelectiveAction.FULL_REVIEW
        else:
            actions[item.index] = SelectiveAction.SHORT_REVIEW
        reasons[item.index].append("BUDGET_ALLOCATED_REVIEW")
        selected.add(item.index)

    for item in optional:
        if item.index in selected:
            continue
        if item.risk > policy.accept_max_risk:
            reasons[item.index].append("BUDGET_CONSTRAINED_ACCEPT")
        else:
            reasons[item.index].append("LOW_OPERATIONAL_RISK")
    if not budget_feasible:
        for item in mandatory:
            reasons[item.index].append("REVIEW_BUDGET_INFEASIBLE")
    return actions, [tuple(values) for values in reasons], budget_feasible


def _review_value(risk: float, cost_profile: CostProfile) -> float:
    full_review_cost = min(cost_profile.full_review, cost_profile.short_review)
    return risk * cost_profile.unsafe_accept - full_review_cost
