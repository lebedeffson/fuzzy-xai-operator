from __future__ import annotations

from itertools import combinations
from math import isclose

from .models import Candidate, Case, Gold


def _oracle_candidates(case: Case) -> tuple[Candidate, ...]:
    allowed = {
        atom_id
        for mutation in case.mutations
        for atom_id in mutation.allowed_inverse_ids
    }
    return tuple(candidate for candidate in case.candidates if candidate.atom_id in allowed)


def _covers(cut: tuple[Candidate, ...]) -> frozenset[str]:
    return frozenset(item for candidate in cut for item in candidate.covers)


def oracle_a(case: Case) -> tuple[float, set[tuple[str, ...]]]:
    candidates = _oracle_candidates(case)
    target = frozenset(case.obligations)
    best = float("inf")
    cuts: set[tuple[str, ...]] = set()
    for size in range(len(candidates) + 1):
        for cut in combinations(candidates, size):
            if not target.issubset(_covers(cut)):
                continue
            cost = sum(item.cost for item in cut)
            ids = tuple(sorted(item.atom_id for item in cut))
            if cost < best and not isclose(cost, best):
                best, cuts = cost, {ids}
            elif isclose(cost, best):
                cuts.add(ids)
    return best, cuts


def oracle_b(case: Case) -> tuple[float, set[tuple[str, ...]]]:
    candidates = _oracle_candidates(case)
    indexes = {value: index for index, value in enumerate(case.obligations)}
    full_mask = (1 << len(indexes)) - 1
    states: dict[int, tuple[float, set[tuple[str, ...]]]] = {0: (0.0, {()})}
    for candidate in candidates:
        coverage = sum(1 << indexes[item] for item in candidate.covers if item in indexes)
        updated = dict(states)
        for mask, (cost, cuts) in states.items():
            new_mask = mask | coverage
            new_cost = cost + candidate.cost
            new_cuts = {tuple(sorted((*cut, candidate.atom_id))) for cut in cuts}
            previous = updated.get(new_mask)
            if previous is None or new_cost < previous[0] and not isclose(new_cost, previous[0]):
                updated[new_mask] = (new_cost, new_cuts)
            elif previous is not None and isclose(new_cost, previous[0]):
                updated[new_mask] = (previous[0], previous[1] | new_cuts)
        states = updated
    return states.get(full_mask, (float("inf"), set()))


def derive_gold(case: Case) -> Gold:
    if not case.repairable:
        return Gold(case.case_id, "NON_REPAIRABLE", (), None, False, None, None)
    cost_a, cuts_a = oracle_a(case)
    cost_b, cuts_b = oracle_b(case)
    if not isclose(cost_a, cost_b) or cuts_a != cuts_b:
        return Gold(
            case.case_id,
            "UNCERTIFIED_SOLVER_DISAGREEMENT",
            (),
            None,
            True,
            None if cost_a == float("inf") else cost_a,
            None if cost_b == float("inf") else cost_b,
        )
    if cost_a == float("inf"):
        return Gold(case.case_id, "INSUFFICIENT_FORMAL_SPECIFICATION", (), None, True, None, None)
    for cut_ids in cuts_a:
        selected = tuple(item for item in _oracle_candidates(case) if item.atom_id in cut_ids)
        if not set(case.obligations).issubset(_covers(selected)):
            raise AssertionError("oracle returned an uncovered cut")
    return Gold(
        case.case_id,
        "CERTIFIED_MULTIPLE_OPTIMA" if len(cuts_a) > 1 else "CERTIFIED_UNIQUE",
        tuple(sorted(cuts_a)),
        cost_a,
        True,
        cost_a,
        cost_b,
    )

