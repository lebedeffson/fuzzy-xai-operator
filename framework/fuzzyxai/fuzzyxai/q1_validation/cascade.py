"""Adaptive A/B/C explanation cascade and matched policy evaluation."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from enum import Enum
from typing import Sequence

import numpy as np


class CascadeLevel(str, Enum):
    A = "A"
    B = "B"
    C = "C"


@dataclass(frozen=True)
class CascadeSignals:
    confidence: float
    required_fields_complete: bool
    distribution_shift: float
    explanation_stability: float
    source_conflict: float
    rare_group: bool
    boundary_score: float

    def __post_init__(self) -> None:
        values = (
            self.confidence,
            self.distribution_shift,
            self.explanation_stability,
            self.source_conflict,
            self.boundary_score,
        )
        if any(not 0.0 <= value <= 1.0 for value in values):
            raise ValueError("cascade signals must be within [0, 1]")


@dataclass(frozen=True)
class CascadePolicy:
    confidence_threshold: float = 0.75
    shift_threshold: float = 0.20
    stability_threshold: float = 0.70
    conflict_threshold: float = 0.20
    boundary_threshold: float = 0.25

    def level(self, signal: CascadeSignals) -> CascadeLevel:
        escalate_b = (
            signal.confidence < self.confidence_threshold
            or not signal.required_fields_complete
            or signal.distribution_shift > self.shift_threshold
            or signal.explanation_stability < self.stability_threshold
            or signal.source_conflict > self.conflict_threshold
            or signal.rare_group
            or signal.boundary_score < self.boundary_threshold
        )
        if not escalate_b:
            return CascadeLevel.A
        escalate_c = (
            not signal.required_fields_complete
            or signal.source_conflict > self.conflict_threshold
            or signal.rare_group
            or signal.distribution_shift > 1.5 * self.shift_threshold
            or signal.explanation_stability < 0.75 * self.stability_threshold
        )
        return CascadeLevel.C if escalate_c else CascadeLevel.B


@dataclass(frozen=True)
class CascadeEvaluation:
    policy_id: str
    n_objects: int
    risk: float
    automatic_coverage: float
    wrong_automatic: int
    critical_wrong_automatic: int
    review: int
    false_block: int
    mean_cost: float
    level_distribution: dict[str, int]

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def cascade_decisions(signals: Sequence[CascadeSignals], policy: CascadePolicy) -> tuple[tuple[CascadeLevel, ...], tuple[str, ...]]:
    levels = tuple(policy.level(item) for item in signals)
    actions: list[str] = []
    for signal, level in zip(signals, levels):
        if level is CascadeLevel.A:
            actions.append("accept")
        elif level is CascadeLevel.B:
            actions.append("review")
        elif not signal.required_fields_complete or signal.source_conflict > policy.conflict_threshold:
            actions.append("block")
        else:
            actions.append("review")
    return levels, tuple(actions)


def evaluate_cascade(
    signals: Sequence[CascadeSignals],
    *,
    predictions: Sequence[int],
    labels: Sequence[int],
    critical: Sequence[bool],
    policy: CascadePolicy | None = None,
    level_costs: tuple[float, float, float] = (1.0, 3.0, 10.0),
    error_costs: tuple[float, float, float, float] = (20.0, 5.0, 1.0, 2.0),
) -> CascadeEvaluation:
    if not signals or not (len(signals) == len(predictions) == len(labels) == len(critical)):
        raise ValueError("cascade inputs must be non-empty and aligned")
    active = policy or CascadePolicy()
    level_values, action_values = cascade_decisions(signals, active)
    levels = list(level_values)
    actions = list(action_values)
    wrong = np.asarray(predictions) != np.asarray(labels)
    auto = np.asarray([item == "accept" for item in actions])
    blocked = np.asarray([item == "block" for item in actions])
    critical_array = np.asarray(critical, dtype=bool)
    wrong_auto = int(np.sum(wrong & auto))
    critical_wrong = int(np.sum(wrong & auto & critical_array))
    false_block = int(np.sum(blocked & ~wrong))
    review = int(sum(item == "review" for item in actions))
    critical_cost, wrong_cost, review_cost, block_cost = error_costs
    decision_risk = (
        critical_cost * critical_wrong
        + wrong_cost * wrong_auto
        + review_cost * review
        + block_cost * false_block
    ) / len(signals)
    computational = sum(level_costs[(CascadeLevel.A, CascadeLevel.B, CascadeLevel.C).index(level)] for level in levels) / len(levels)
    return CascadeEvaluation(
        policy_id="adaptive_ABC",
        n_objects=len(signals),
        risk=float(decision_risk),
        automatic_coverage=float(np.mean(auto)),
        wrong_automatic=wrong_auto,
        critical_wrong_automatic=critical_wrong,
        review=review,
        false_block=false_block,
        mean_cost=float(computational),
        level_distribution={level.value: levels.count(level) for level in CascadeLevel},
    )
