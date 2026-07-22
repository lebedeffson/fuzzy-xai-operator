from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations
from typing import Iterable


@dataclass(frozen=True)
class CutOracleResult:
    optimal_cuts: tuple[tuple[str, ...], ...]
    optimal_cost: float
    explored_subsets: int


def enumerate_optimal_cuts(
    broken_paths: Iterable[Iterable[str]],
    costs: dict[str, float],
) -> CutOracleResult:
    """Exhaustively enumerate every minimum-cost hitting set.

    This implementation intentionally does not call the evaluated H10 solver.
    Gold graphs are kept small enough for exhaustive adjudication.
    """
    paths = tuple(frozenset(path) for path in broken_paths if tuple(path))
    if not paths:
        return CutOracleResult(((),), 0.0, 1)
    path_nodes = frozenset().union(*paths)
    repairable = tuple(sorted(node for node in path_nodes if node in costs))
    nodes = repairable or tuple(sorted(path_nodes))
    if any(not set(nodes).intersection(path) for path in paths):
        raise ValueError("a broken path has no independently registered repairable element")
    best_cost = float("inf")
    best: list[tuple[str, ...]] = []
    explored = 0
    for size in range(1, len(nodes) + 1):
        for candidate in combinations(nodes, size):
            explored += 1
            chosen = frozenset(candidate)
            if not all(chosen.intersection(path) for path in paths):
                continue
            cost = float(sum(costs.get(node, 1.0) for node in candidate))
            if cost < best_cost - 1e-12:
                best_cost = cost
                best = [tuple(sorted(candidate))]
            elif abs(cost - best_cost) <= 1e-12:
                best.append(tuple(sorted(candidate)))
    if not best:
        raise ValueError("no diagnostic cut covers every broken path")
    return CutOracleResult(tuple(sorted(set(best))), best_cost, explored)
