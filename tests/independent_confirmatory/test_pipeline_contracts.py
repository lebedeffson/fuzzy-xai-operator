from __future__ import annotations

import numpy as np

from experiments.independent_confirmatory.formative import assign_actions, policy_scores


def _rows(count: int = 1000) -> dict[str, np.ndarray]:
    values = np.linspace(0.0, 1.0, count, endpoint=False)
    return {
        "object_ids": np.asarray([f"object-{index}" for index in range(count)]),
        "predictive_risk": values,
        "entropy": values[::-1].copy(),
        "margin_risk": values,
        "route_risk": (values < 0.04).astype(float),
        "repairable_fault": (values < 0.035),
        "irreparable_fault": (values >= 0.035) & (values < 0.04),
        "explanation_risk": values,
        "shift_risk": values,
    }


def test_full_policy_uses_repair_before_block_and_respects_review_equivalent_budget() -> None:
    rows = _rows()
    actions = assign_actions(rows, "full_hierarchical_fuzzyxai", budget=0.20)
    assert np.mean(actions == "block") == 0.005
    assert np.mean(actions == "repair_then_retry") == 0.035
    assert np.mean(np.isin(actions, ("short_review", "full_review", "repair_then_retry"))) == 0.20
    assert not np.any(actions[rows["repairable_fault"]] == "block")


def test_policy_scores_are_observable_and_aligned() -> None:
    rows = _rows(120)
    scores = policy_scores(rows)
    assert "full_hierarchical_fuzzyxai" in scores
    assert all(len(value) == 120 and np.all(np.isfinite(value)) for value in scores.values())
