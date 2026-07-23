from __future__ import annotations

from math import isclose
from time import perf_counter

from .contracts import DiagnosticCut, RepairCostModel, RouteGraph, ValidationObligation, ValidationResult


class MinimalDiagnosticCutFinder:
    def __init__(self, *, exact_atom_limit: int = 28, equivalent_limit: int = 256) -> None:
        self.exact_atom_limit = exact_atom_limit
        self.equivalent_limit = equivalent_limit

    def find(
        self,
        graph: RouteGraph,
        validation: ValidationResult,
        costs: RepairCostModel | None = None,
    ) -> DiagnosticCut:
        del graph
        started = perf_counter()
        cost_model = costs or RepairCostModel()
        repairable = tuple(item for item in validation.obligations if item.repairable and item.candidate_atoms)
        impossible = tuple(item.obligation_id for item in validation.obligations if not item.candidate_atoms)
        atoms = tuple(sorted({atom for item in repairable for atom in item.candidate_atoms}))
        if not validation.obligations:
            return DiagnosticCut((), (), 0.0, True, "none", (), (), ((),), (perf_counter() - started) * 1000)
        if not atoms:
            return DiagnosticCut(
                (),
                (),
                0.0,
                False,
                "unavailable",
                (),
                tuple(item.obligation_id for item in validation.obligations),
                (),
                (perf_counter() - started) * 1000,
            )
        if len(atoms) <= self.exact_atom_limit:
            solutions, best_cost = self._exact(repairable, cost_model)
            selected = solutions[0]
            optimal = True
            solver = "branch_and_bound"
        else:
            selected = self._greedy(repairable, cost_model)
            solutions = (selected,)
            best_cost = sum(cost_model.cost(atom) for atom in selected)
            optimal = False
            solver = "greedy_weighted_hitting_set"
        covered = self._covered(selected, repairable)
        uncovered = tuple(
            item.obligation_id for item in validation.obligations if item.obligation_id not in covered
        )
        uncovered = tuple(dict.fromkeys((*uncovered, *impossible)))
        cut = DiagnosticCut(
            defect_atoms=selected,
            affected_nodes=tuple(sorted({self._atom_subject(atom) for atom in selected})),
            total_cost=float(best_cost),
            optimal=optimal and not uncovered,
            solver=solver,
            covered_obligations=tuple(sorted(covered)),
            uncovered_obligations=tuple(sorted(uncovered)),
            equivalent_optimal_cuts=solutions if optimal else (),
            runtime_ms=(perf_counter() - started) * 1000,
        )
        verify_cut(cut, validation.obligations, cost_model)
        return cut

    def _exact(
        self,
        obligations: tuple[ValidationObligation, ...],
        costs: RepairCostModel,
    ) -> tuple[tuple[tuple[str, ...], ...], float]:
        best_cost = float("inf")
        solutions: set[tuple[str, ...]] = set()

        def search(remaining: tuple[ValidationObligation, ...], chosen: frozenset[str], current: float) -> None:
            nonlocal best_cost
            if current > best_cost or (isclose(current, best_cost) and len(solutions) >= self.equivalent_limit):
                return
            if not remaining:
                candidate = tuple(sorted(chosen))
                if current < best_cost and not isclose(current, best_cost):
                    best_cost = current
                    solutions.clear()
                if isclose(current, best_cost):
                    solutions.add(candidate)
                return
            obligation = min(remaining, key=lambda item: (len(item.candidate_atoms), item.obligation_id))
            for atom in sorted(obligation.candidate_atoms, key=lambda value: (costs.cost(value), value)):
                added = 0.0 if atom in chosen else costs.cost(atom)
                if current + added > best_cost:
                    continue
                next_remaining = tuple(item for item in remaining if atom not in item.candidate_atoms)
                search(next_remaining, chosen | {atom}, current + added)

        search(obligations, frozenset(), 0.0)
        if not solutions:
            raise ValueError("no repairable diagnostic cut covers every repairable obligation")
        ordered = tuple(sorted(solutions, key=lambda value: (len(value), value)))
        return ordered, best_cost

    @staticmethod
    def _greedy(
        obligations: tuple[ValidationObligation, ...],
        costs: RepairCostModel,
    ) -> tuple[str, ...]:
        remaining = list(obligations)
        chosen: list[str] = []
        while remaining:
            atoms = {atom for item in remaining for atom in item.candidate_atoms}
            if not atoms:
                break
            selected = min(
                atoms,
                key=lambda atom: (
                    costs.cost(atom) / sum(atom in item.candidate_atoms for item in remaining),
                    atom,
                ),
            )
            chosen.append(selected)
            remaining = [item for item in remaining if selected not in item.candidate_atoms]
        return tuple(sorted(chosen))

    @staticmethod
    def _covered(
        selected: tuple[str, ...],
        obligations: tuple[ValidationObligation, ...],
    ) -> set[str]:
        chosen = set(selected)
        return {
            item.obligation_id
            for item in obligations
            if chosen.intersection(item.candidate_atoms)
        }

    @staticmethod
    def _atom_subject(atom: str) -> str:
        parts = atom.split("/")
        location = next((part for part in parts if part.startswith(("node:", "edge:"))), atom)
        return location.split(":", 1)[1] if ":" in location else location


def verify_cut(
    cut: DiagnosticCut,
    obligations: tuple[ValidationObligation, ...],
    costs: RepairCostModel | None = None,
) -> None:
    cost_model = costs or RepairCostModel()
    selected = set(cut.defect_atoms)
    expected_covered = {
        item.obligation_id
        for item in obligations
        if selected.intersection(item.candidate_atoms)
    }
    expected_uncovered = {item.obligation_id for item in obligations} - expected_covered
    if expected_covered != set(cut.covered_obligations):
        raise ValueError("diagnostic cut covered-obligation list is inconsistent")
    if expected_uncovered != set(cut.uncovered_obligations):
        raise ValueError("diagnostic cut leaves an unreported obligation")
    expected_cost = sum(cost_model.cost(atom) for atom in cut.defect_atoms)
    if not isclose(expected_cost, cut.total_cost):
        raise ValueError("diagnostic cut cost is inconsistent")
    if cut.optimal and cut.uncovered_obligations:
        raise ValueError("an incomplete diagnostic cut cannot be marked optimal")
    if cut.optimal and cut.defect_atoms not in cut.equivalent_optimal_cuts:
        raise ValueError("selected exact cut is absent from equivalent optimal cuts")
