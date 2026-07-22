"""Development-only matching of candidate rules to comparable controls."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

from .conditional import ConditionalRule


@dataclass(frozen=True)
class MatchedRuleSet:
    candidate: ConditionalRule
    controls: tuple[ConditionalRule, ...]
    distances: tuple[float, ...]
    selected_on: str = "development"


def match_controls(candidate: ConditionalRule, rules: Sequence[ConditionalRule], *, count: int = 5) -> MatchedRuleSet:
    if count < 5:
        raise ValueError("confirmatory matching requires at least five controls")
    eligible = [rule for rule in rules if rule.rule_id != candidate.rule_id and rule.predicted_class == candidate.predicted_class]
    ranked = sorted(((rule, _distance(candidate, rule)) for rule in eligible), key=lambda item: (item[1], item[0].rule_id))
    if len(ranked) < count:
        raise ValueError("insufficient matched controls")
    selected = ranked[:count]
    return MatchedRuleSet(candidate, tuple(item[0] for item in selected), tuple(float(item[1]) for item in selected))


def eligible_candidate(rule: ConditionalRule, *, unique_coverage: float, direction_stable: bool, subgroup_leakage: bool) -> bool:
    return (
        rule.bootstrap_stability >= 0.80
        and rule.support >= 0.05
        and rule.redundancy <= 0.50
        and unique_coverage > 0.0
        and direction_stable
        and not subgroup_leakage
    )


def specific_effect(candidate_effect: float, control_effects: Sequence[float]) -> float:
    if len(control_effects) < 5:
        raise ValueError("specific effect requires at least five matched controls")
    ordered = sorted(float(value) for value in control_effects)
    middle = len(ordered) // 2
    median = ordered[middle] if len(ordered) % 2 else (ordered[middle - 1] + ordered[middle]) / 2.0
    return float(candidate_effect - median)


def _distance(left: ConditionalRule, right: ConditionalRule) -> float:
    return (
        abs(left.support - right.support)
        + abs(left.redundancy - right.redundancy)
        + abs(left.length - right.length) / max(1, left.length)
        + abs(left.depth - right.depth) / max(1, left.depth)
        + abs(left.subgroup_size - right.subgroup_size) / max(1, left.subgroup_size)
    )
