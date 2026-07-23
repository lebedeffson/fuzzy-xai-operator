"""Independent H10-C3 R4 Gold oracle.

This module intentionally has no imports from FuzzyXAI or the evaluated R4
method package. It operates on plain dictionaries derived from the private
mutation transaction and the public candidate matrix.
"""

from __future__ import annotations

from decimal import Decimal
from itertools import combinations


def _cost(candidate: dict[str, object]) -> Decimal:
    return Decimal(str(candidate["cost"]))


def _coverage(
    selected: tuple[dict[str, object], ...],
) -> frozenset[str]:
    return frozenset(
        obligation
        for candidate in selected
        for obligation in candidate["covers"]
    )


def exhaustive_oracle(
    candidates: tuple[dict[str, object], ...],
    obligations: tuple[str, ...],
) -> tuple[Decimal, set[tuple[str, ...]]]:
    required = frozenset(obligations)
    best = Decimal("Infinity")
    cuts: set[tuple[str, ...]] = set()
    for size in range(len(candidates) + 1):
        for selected in combinations(candidates, size):
            if not required.issubset(_coverage(selected)):
                continue
            cost = sum((_cost(item) for item in selected), start=Decimal(0))
            candidate_ids = tuple(
                sorted(str(item["candidate_id"]) for item in selected)
            )
            if cost < best:
                best = cost
                cuts = {candidate_ids}
            elif cost == best:
                cuts.add(candidate_ids)
    return best, cuts


def dynamic_oracle(
    candidates: tuple[dict[str, object], ...],
    obligations: tuple[str, ...],
) -> tuple[Decimal, set[tuple[str, ...]]]:
    indexes = {obligation: index for index, obligation in enumerate(obligations)}
    full_mask = (1 << len(indexes)) - 1
    states: dict[int, tuple[Decimal, set[tuple[str, ...]]]] = {
        0: (Decimal(0), {()})
    }
    for candidate in candidates:
        mask = sum(
            1 << indexes[obligation]
            for obligation in candidate["covers"]
            if obligation in indexes
        )
        updated = dict(states)
        for prior_mask, (prior_cost, cuts) in states.items():
            next_mask = prior_mask | mask
            next_cost = prior_cost + _cost(candidate)
            next_cuts = {
                tuple(
                    sorted(
                        (*cut, str(candidate["candidate_id"])),
                    )
                )
                for cut in cuts
            }
            previous = updated.get(next_mask)
            if previous is None or next_cost < previous[0]:
                updated[next_mask] = (next_cost, next_cuts)
            elif next_cost == previous[0]:
                updated[next_mask] = (
                    previous[0],
                    previous[1] | next_cuts,
                )
        states = updated
    return states.get(full_mask, (Decimal("Infinity"), set()))


def derive_gold(
    *,
    private_record: dict[str, object],
    public_candidates: tuple[dict[str, object], ...],
    obligations: tuple[str, ...],
    repairable: bool,
) -> dict[str, object]:
    if not repairable:
        return {
            "status": "NON_REPAIRABLE",
            "optimal_cuts": (),
            "optimal_cost": None,
            "solver_a_cost": None,
            "solver_b_cost": None,
            "repairable": False,
        }
    mutation = dict(private_record["mutation"])
    allowed = set(mutation["reverse_candidate_ids"])
    candidates = tuple(
        candidate
        for candidate in public_candidates
        if candidate["candidate_id"] in allowed
    )
    cost_a, cuts_a = exhaustive_oracle(candidates, obligations)
    cost_b, cuts_b = dynamic_oracle(candidates, obligations)
    if cost_a != cost_b or cuts_a != cuts_b:
        return {
            "status": "UNCERTIFIED_SOLVER_DISAGREEMENT",
            "optimal_cuts": (),
            "optimal_cost": None,
            "solver_a_cost": None if cost_a.is_infinite() else float(cost_a),
            "solver_b_cost": None if cost_b.is_infinite() else float(cost_b),
            "repairable": True,
        }
    if cost_a.is_infinite():
        return {
            "status": "INSUFFICIENT_FORMAL_SPECIFICATION",
            "optimal_cuts": (),
            "optimal_cost": None,
            "solver_a_cost": None,
            "solver_b_cost": None,
            "repairable": True,
        }
    return {
        "status": (
            "CERTIFIED_MULTIPLE_OPTIMA"
            if len(cuts_a) > 1
            else "CERTIFIED_UNIQUE"
        ),
        "optimal_cuts": tuple(sorted(cuts_a)),
        "optimal_cost": float(cost_a),
        "solver_a_cost": float(cost_a),
        "solver_b_cost": float(cost_b),
        "repairable": True,
    }
