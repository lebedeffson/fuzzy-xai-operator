from __future__ import annotations

from dataclasses import dataclass
from time import perf_counter

from .models import DiagnosticCutResult


@dataclass
class DiagnosticCutSolver:
    exact_node_limit: int = 24

    def solve(self, invalid_paths: tuple[frozenset[str], ...], costs: dict[str, float]) -> DiagnosticCutResult:
        started = perf_counter()
        paths = tuple(path for path in invalid_paths if path)
        if not paths:
            return DiagnosticCutResult((), 0.0, True, "none", (perf_counter() - started) * 1000.0, 0)
        nodes = tuple(sorted(frozenset().union(*paths)))
        if len(nodes) <= self.exact_node_limit:
            selected = self._branch_and_bound(paths, costs)
            optimal, solver = True, "branch_and_bound"
        else:
            selected = self._greedy(paths, costs)
            optimal, solver = False, "greedy_weighted_hitting_set"
        elapsed = (perf_counter() - started) * 1000.0
        covered = sum(bool(set(selected) & path) for path in paths)
        return DiagnosticCutResult(
            tuple(sorted(selected)),
            float(sum(costs.get(node, 1.0) for node in selected)),
            optimal,
            solver,
            elapsed,
            covered,
        )

    @staticmethod
    def _branch_and_bound(paths: tuple[frozenset[str], ...], costs: dict[str, float]) -> tuple[str, ...]:
        best_cost = float("inf")
        best: tuple[str, ...] | None = None

        def search(remaining: tuple[frozenset[str], ...], chosen: tuple[str, ...], cost: float) -> None:
            nonlocal best_cost, best
            if cost >= best_cost:
                return
            if not remaining:
                candidate = tuple(sorted(chosen))
                if cost < best_cost or (cost == best_cost and (best is None or candidate < best)):
                    best_cost, best = cost, candidate
                return
            path = min(remaining, key=lambda item: (len(item), tuple(sorted(item))))
            for node in sorted(path, key=lambda item: (costs.get(item, 1.0), item)):
                search(tuple(other for other in remaining if node not in other), chosen + (node,), cost + costs.get(node, 1.0))

        search(paths, (), 0.0)
        if best is None:
            raise ValueError("no diagnostic cut covers every invalid path")
        return best

    @staticmethod
    def _greedy(paths: tuple[frozenset[str], ...], costs: dict[str, float]) -> tuple[str, ...]:
        remaining = list(paths)
        chosen: list[str] = []
        while remaining:
            nodes = sorted(frozenset().union(*remaining))
            node = max(nodes, key=lambda item: (sum(item in path for path in remaining) / max(costs.get(item, 1.0), 1e-12), item))
            chosen.append(node)
            remaining = [path for path in remaining if node not in path]
        return tuple(chosen)
