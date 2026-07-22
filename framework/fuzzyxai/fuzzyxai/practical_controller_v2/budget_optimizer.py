"""Global review-budget allocation by marginal expected-loss reduction."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

from fuzzyxai.practical_controller import HardGuardStatus
from fuzzyxai.selective_observer import SelectiveAction

from .expected_loss import ExpectedActionLosses


@dataclass(frozen=True)
class BudgetCandidate:
    index: int
    losses: ExpectedActionLosses
    hard_guard_status: HardGuardStatus

    @property
    def preferred_review(self) -> SelectiveAction:
        return SelectiveAction.SHORT_REVIEW if self.losses.short_review <= self.losses.full_review else SelectiveAction.FULL_REVIEW

    @property
    def marginal_review_benefit(self) -> float:
        return self.losses.accept - min(self.losses.short_review, self.losses.full_review)


def optimize_review_budget(
    candidates: Sequence[BudgetCandidate],
    *,
    review_budget: float,
) -> tuple[tuple[SelectiveAction, ...], bool]:
    if not candidates or not 0.0 <= review_budget <= 1.0:
        raise ValueError("candidates and a valid review budget are required")
    actions = [SelectiveAction.ACCEPT for _ in candidates]
    mandatory = []
    for candidate in candidates:
        if candidate.hard_guard_status is HardGuardStatus.BLOCKED:
            actions[candidate.index] = SelectiveAction.BLOCK
        elif candidate.hard_guard_status is HardGuardStatus.REVIEW_REQUIRED:
            mandatory.append(candidate)
    capacity = int(review_budget * len(candidates) + 1e-12)
    feasible = len(mandatory) <= capacity
    for candidate in mandatory:
        actions[candidate.index] = SelectiveAction.FULL_REVIEW
    remaining = max(0, capacity - len(mandatory))
    optional = [
        candidate
        for candidate in candidates
        if candidate.hard_guard_status is HardGuardStatus.CERTIFIED and candidate.marginal_review_benefit > 0.0
    ]
    optional.sort(key=lambda item: (-item.marginal_review_benefit, item.index))
    for candidate in optional[:remaining]:
        actions[candidate.index] = candidate.preferred_review
    return tuple(actions), feasible
