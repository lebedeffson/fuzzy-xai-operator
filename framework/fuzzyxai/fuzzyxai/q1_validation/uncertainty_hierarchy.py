"""Q1 comparison of fixed, adaptive and refusal uncertainty modes."""

from __future__ import annotations

from typing import Mapping, Sequence

from fuzzyxai.experiments.uncertainty_selection import evaluate_selection_modes


def compare_uncertainty_modes(
    profiles: Sequence[Sequence[str]],
    *,
    action_risks: Mapping[str, Sequence[float]],
    epsilon_risk: float = 0.02,
) -> dict[str, object]:
    result = evaluate_selection_modes(profiles, epsilon=epsilon_risk, action_risks=action_risks)
    rows = list(result["rows"])
    adaptive = next(row for row in rows if row["mode"] == "adaptive")
    full = next(row for row in rows if row["mode"] == "always_FML")
    result["complexity_reduction_vs_fml"] = float(full["mean_complexity"]) - float(adaptive["mean_complexity"])
    result["claim_allowed"] = bool(result["non_inferior_to_fml"] and result["complexity_reduction_vs_fml"] > 0.0)
    result["allowed_wording"] = (
        "Adaptive selection preserves risk within the preregistered margin and reduces representation complexity."
        if result["claim_allowed"]
        else "Adaptive representation utility was not established under the preregistered criteria."
    )
    result["forbidden_wording"] = "Adaptive selection is superior on every criterion."
    return result
