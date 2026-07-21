"""Controlled uncertainty hierarchy experiment without hidden oracle access."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Mapping, Sequence


REPRESENTATION_COVERAGE: dict[str, frozenset[str]] = {
    "F0": frozenset({"aleatoric"}),
    "Fint": frozenset({"aleatoric", "interval_imprecision"}),
    "NAS": frozenset({"aleatoric", "interval_imprecision", "source_conflict", "incomplete_trace"}),
    "FML": frozenset(
        {
            "aleatoric",
            "interval_imprecision",
            "source_conflict",
            "incomplete_trace",
            "distribution_shift",
            "temporal_instability",
            "counterfactual_instability",
            "user_disagreement",
        }
    ),
}

REPRESENTATION_COMPLEXITY = {"F0": 1.0, "Fint": 2.0, "NAS": 4.0, "FML": 8.0, "diagnostic_refusal": 0.5}


@dataclass(frozen=True)
class SelectionResult:
    representation: str
    covered: bool
    undercovered_types: tuple[str, ...]
    complexity: float
    reason: str


def select_minimal_representation(required_types: Iterable[str]) -> SelectionResult:
    required = frozenset(str(item) for item in required_types)
    for name in ("F0", "Fint", "NAS", "FML"):
        missing = required - REPRESENTATION_COVERAGE[name]
        if not missing:
            return SelectionResult(name, True, (), REPRESENTATION_COMPLEXITY[name], "minimal covering representation")
    return SelectionResult(
        "diagnostic_refusal",
        False,
        tuple(sorted(required)),
        REPRESENTATION_COMPLEXITY["diagnostic_refusal"],
        "uncertainty profile is outside the certified hierarchy",
    )


def evaluate_selection_modes(
    profiles: Sequence[Sequence[str]],
    *,
    epsilon: float,
    action_risks: Mapping[str, Sequence[float]],
) -> dict[str, object]:
    modes = ("F0", "Fint", "NAS", "FML")
    rows: list[dict[str, object]] = []
    adaptive_counts = {name: 0 for name in (*modes, "diagnostic_refusal")}
    adaptive_complexity = 0.0
    adaptive_undercoverage = 0
    for profile in profiles:
        selected = select_minimal_representation(profile)
        adaptive_counts[selected.representation] += 1
        adaptive_complexity += selected.complexity
        adaptive_undercoverage += int(not selected.covered)
    for mode in modes:
        undercoverage = sum(not set(profile).issubset(REPRESENTATION_COVERAGE[mode]) for profile in profiles)
        rows.append(
            {
                "mode": f"always_{mode}",
                "coverage": 1.0 - undercoverage / len(profiles),
                "undercoverage": undercoverage / len(profiles),
                "mean_complexity": REPRESENTATION_COMPLEXITY[mode],
                "mean_risk": sum(action_risks[mode]) / len(action_risks[mode]),
            }
        )
    adaptive_risk = sum(action_risks["adaptive"]) / len(action_risks["adaptive"])
    fml_risk = sum(action_risks["FML"]) / len(action_risks["FML"])
    fml_fraction = adaptive_counts["FML"] / len(profiles)
    rows.append(
        {
            "mode": "adaptive",
            "coverage": 1.0 - adaptive_undercoverage / len(profiles),
            "undercoverage": adaptive_undercoverage / len(profiles),
            "mean_complexity": adaptive_complexity / len(profiles),
            "mean_risk": adaptive_risk,
        }
    )
    return {
        "rows": rows,
        "adaptive_distribution": adaptive_counts,
        "adaptive_fml_fraction": fml_fraction,
        "non_inferiority_epsilon": epsilon,
        "non_inferior_to_fml": adaptive_risk <= fml_risk + epsilon,
        "practical_hierarchy_claim_allowed": fml_fraction <= 0.9,
        "claim_block_reason": "adaptive_selects_fml_above_90_percent" if fml_fraction > 0.9 else None,
    }
