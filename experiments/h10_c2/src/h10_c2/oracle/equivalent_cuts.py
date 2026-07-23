from __future__ import annotations

from itertools import combinations
from math import isclose
from typing import Iterable


def enumerate_optimal_cuts(
    obligations: Iterable[Iterable[str]],
    costs: dict[str, float],
    *,
    max_atoms: int = 18,
) -> tuple[tuple[tuple[str, ...], ...], float, int]:
    paths = tuple(frozenset(path) for path in obligations)
    if not paths:
        return ((),), 0.0, 1
    atoms = tuple(sorted(frozenset().union(*paths)))
    if len(atoms) > max_atoms:
        raise OverflowError("independent exhaustive oracle atom limit exceeded")
    if any(not path for path in paths):
        raise ValueError("an obligation has no repair candidate")
    best_cost = float("inf")
    best: set[tuple[str, ...]] = set()
    explored = 0
    for size in range(1, len(atoms) + 1):
        for candidate in combinations(atoms, size):
            explored += 1
            chosen = frozenset(candidate)
            if not all(chosen.intersection(path) for path in paths):
                continue
            total = float(sum(costs[atom] for atom in candidate))
            if total < best_cost and not isclose(total, best_cost):
                best_cost = total
                best = {tuple(sorted(candidate))}
            elif isclose(total, best_cost):
                best.add(tuple(sorted(candidate)))
    if not best:
        raise ValueError("no feasible diagnostic cut")
    return tuple(sorted(best)), best_cost, explored

