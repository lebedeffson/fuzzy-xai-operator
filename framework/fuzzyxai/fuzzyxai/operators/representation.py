from __future__ import annotations

from collections.abc import Sequence

from fuzzyxai.selection.pareto_selector import Candidate, select_minimal_sufficient


def select_representation_class(profile: set[str], candidates: Sequence[Candidate], mode: str = "audit"):
    """Select the minimal sufficient representation through the core Pareto selector."""

    return select_minimal_sufficient(profile, candidates, mode=mode)
