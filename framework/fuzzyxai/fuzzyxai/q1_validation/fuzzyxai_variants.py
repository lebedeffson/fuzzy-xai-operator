"""Predeclared FuzzyXAI variants used in paired and cascade analyses."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class FuzzyXAIVariant:
    variant_id: str
    label: str
    channels: tuple[str, ...]
    adaptive: bool = False


VARIANTS: tuple[FuzzyXAIVariant, ...] = (
    FuzzyXAIVariant("FX0", "local explainer without FuzzyXAI", ("local_explanation",)),
    FuzzyXAIVariant("FX1", "FuzzyXAI with local explainer only", ("local_explanation", "claim_graph")),
    FuzzyXAIVariant("FX2", "FuzzyXAI with provenance and incompleteness", ("local_explanation", "claim_graph", "provenance", "missingness")),
    FuzzyXAIVariant("FX3", "FuzzyXAI without training history", ("local_explanation", "claim_graph", "provenance", "missingness", "stability")),
    FuzzyXAIVariant("FX4", "full FuzzyXAI", ("local_explanation", "claim_graph", "provenance", "missingness", "stability", "training_history", "uncertainty", "audit")),
    FuzzyXAIVariant("FX5", "adaptive cascade", ("level_A", "level_B", "level_C"), adaptive=True),
)


def get_variant(variant_id: str) -> FuzzyXAIVariant:
    for item in VARIANTS:
        if item.variant_id == variant_id:
            return item
    raise KeyError(variant_id)
