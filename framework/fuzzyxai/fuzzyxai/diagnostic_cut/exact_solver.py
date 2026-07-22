"""Exact minimum-cost hitting-set solver for registered failure paths."""

from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations

from .graph import DiagnosticGraph


@dataclass(frozen=True)
class MinimalDiagnosticCut:
    contracts: tuple[str, ...]
    total_repair_cost: float
    exact: bool
    fault_sources: tuple[str, ...]


def solve_exact(graph: DiagnosticGraph, *, maximum_contracts: int = 22) -> MinimalDiagnosticCut:
    contracts = sorted(graph.contracts)
    if not contracts:
        return MinimalDiagnosticCut((), 0.0, True, ())
    if len(contracts) > maximum_contracts:
        raise ValueError("exact diagnostic cut exceeds configured search width")
    best: tuple[float, tuple[str, ...]] | None = None
    for size in range(1, len(contracts) + 1):
        for candidate in combinations(contracts, size):
            chosen = frozenset(candidate)
            if not all(chosen & path for path in graph.failure_paths):
                continue
            score = sum(graph.cost(item) for item in candidate)
            if best is None or (score, candidate) < best:
                best = (score, candidate)
        if best is not None and len(best[1]) == size:
            break
    assert best is not None
    sources = dict(graph.fault_sources)
    return MinimalDiagnosticCut(best[1], float(best[0]), True, tuple(sorted({sources[item] for item in best[1]})))
