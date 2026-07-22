"""Deterministic greedy approximation for wide diagnostic-cut problems."""

from __future__ import annotations

from .exact_solver import MinimalDiagnosticCut
from .graph import DiagnosticGraph


def solve_approximate(graph: DiagnosticGraph) -> MinimalDiagnosticCut:
    remaining = list(graph.failure_paths)
    selected: list[str] = []
    while remaining:
        candidates = sorted(frozenset().union(*remaining))
        choice = max(candidates, key=lambda item: (sum(item in path for path in remaining) / max(graph.cost(item), 1e-12), item))
        selected.append(choice)
        remaining = [path for path in remaining if choice not in path]
    sources = dict(graph.fault_sources)
    return MinimalDiagnosticCut(
        tuple(sorted(selected)),
        float(sum(graph.cost(item) for item in selected)),
        False,
        tuple(sorted({sources[item] for item in selected})),
    )
