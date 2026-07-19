"""Matched rule-ablation analysis without best-result seed selection."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from math import inf
from typing import Sequence

from fuzzyxai.experiments.statistics import paired_summary


@dataclass(frozen=True)
class RuleDescriptor:
    rule_id: str
    coverage: float
    subgroup_coverage: float
    exclusivity: float
    redundancy: float
    activation_frequency: float
    depth: int
    confidence: float
    output_class: str


@dataclass(frozen=True)
class AblationPair:
    fold: int
    seed: int
    selected_rule: str
    matched_rule: str
    selected_delta: float
    matched_delta: float
    subgroup: str

    @property
    def specific_effect(self) -> float:
        return self.selected_delta - self.matched_delta


def select_matched_random_rule(selected: RuleDescriptor, candidates: Sequence[RuleDescriptor]) -> RuleDescriptor:
    eligible = [item for item in candidates if item.rule_id != selected.rule_id and item.output_class == selected.output_class]
    if not eligible:
        raise ValueError("no matched random rule candidate")

    def distance(item: RuleDescriptor) -> tuple[float, str]:
        value = (
            abs(item.coverage - selected.coverage)
            + abs(item.activation_frequency - selected.activation_frequency)
            + 0.1 * abs(item.depth - selected.depth)
        )
        return value if value < inf else inf, item.rule_id

    return min(eligible, key=distance)


def summarize_ablation(pairs: Sequence[AblationPair], *, seed: int = 4201) -> dict[str, object]:
    if len(pairs) < 50:
        raise ValueError("Q1 rule ablation requires at least 50 paired comparisons")
    selected = [item.selected_delta for item in pairs]
    matched = [item.matched_delta for item in pairs]
    statistic = paired_summary(matched, selected, seed=seed)
    return {
        "n_pairs": len(pairs),
        "folds": len({item.fold for item in pairs}),
        "seeds": len({item.seed for item in pairs}),
        "specific_effect": asdict(statistic),
        "interpretation": (
            "context_dependent_effect_candidate"
            if statistic.confidence_interval_95[0] > 0.0
            else "local_diagnostic_without_general_predictive_claim"
        ),
    }
