from __future__ import annotations

from itertools import product
from math import isclose

from .contracts import DefectAtom, OptimalCutSet, RepairCostModel, RouteGraph, ValidationObligation
from .cut_verification import verify_cut_elements


class ExactMinimalCutSolver:
    def __init__(self, *, equivalent_limit: int = 512) -> None:
        self.equivalent_limit = equivalent_limit

    def solve(
        self,
        graph: RouteGraph,
        obligations: tuple[ValidationObligation, ...],
        atoms: tuple[DefectAtom, ...],
        costs: RepairCostModel | None = None,
    ) -> OptimalCutSet:
        cost_model = costs or RepairCostModel()
        if not verify_cut_elements(graph, atoms):
            raise ValueError("exact solver received an atom outside the route graph")
        relevant = tuple(item for item in obligations if item.repairable and item.candidate_atoms)
        if not relevant:
            return OptimalCutSet(((),), 0.0, True, False, 1, True)
        components = self._obligation_components(relevant)
        if len(components) > 1:
            component_results = [
                self.solve(
                    graph,
                    component,
                    tuple(
                        atom
                        for atom in atoms
                        if any(atom in obligation.candidate_atoms for obligation in component)
                    ),
                    cost_model,
                )
                for component in components
            ]
            lower_bound_count = 1
            for result in component_results:
                lower_bound_count *= result.lower_bound_count
            combined: set[tuple[DefectAtom, ...]] = set()
            truncated = any(result.truncated for result in component_results)
            for selection in product(*(result.cuts for result in component_results)):
                candidate = tuple(
                    sorted(
                        {atom for cut in selection for atom in cut},
                        key=lambda atom: atom.key,
                    )
                )
                if len(combined) < self.equivalent_limit:
                    combined.add(candidate)
                elif candidate not in combined:
                    truncated = True
            ordered = tuple(
                sorted(combined, key=lambda cut: (len(cut), tuple(atom.key for atom in cut)))
            )
            return OptimalCutSet(
                cuts=ordered,
                optimal_cost=sum(result.optimal_cost for result in component_results),
                enumeration_complete=all(
                    result.enumeration_complete for result in component_results
                )
                and not truncated,
                truncated=truncated,
                lower_bound_count=lower_bound_count,
                optimality_proven=all(result.optimality_proven for result in component_results),
            )
        prepared_atoms = self._remove_dominated(relevant, atoms, cost_model)
        obligation_index = {item.obligation_id: index for index, item in enumerate(relevant)}
        full_mask = (1 << len(relevant)) - 1
        masks = {
            atom: sum(
                1 << obligation_index[item.obligation_id]
                for item in relevant
                if atom in item.candidate_atoms
            )
            for atom in prepared_atoms
        }
        masks = {atom: mask for atom, mask in masks.items() if mask}
        if not masks:
            raise ValueError("no exact cut covers the repairable obligations")
        best_cost = float("inf")
        solutions: set[tuple[DefectAtom, ...]] = set()
        truncated = False
        memo: dict[int, float] = {}

        def lower_bound(mask: int) -> float:
            uncovered = [
                item
                for index, item in enumerate(relevant)
                if not mask & (1 << index)
            ]
            if not uncovered:
                return 0.0
            return max(
                min(cost_model.cost(atom) for atom in item.candidate_atoms if atom in masks)
                for item in uncovered
            )

        def search(mask: int, chosen: tuple[DefectAtom, ...], current_cost: float) -> None:
            nonlocal best_cost, truncated
            if current_cost + lower_bound(mask) > best_cost and not isclose(
                current_cost + lower_bound(mask),
                best_cost,
            ):
                return
            previous = memo.get(mask)
            if previous is not None and current_cost > previous and not isclose(current_cost, previous):
                return
            memo[mask] = min(previous, current_cost) if previous is not None else current_cost
            if mask == full_mask:
                candidate = tuple(sorted(chosen, key=lambda atom: atom.key))
                if current_cost < best_cost and not isclose(current_cost, best_cost):
                    best_cost = current_cost
                    solutions.clear()
                    truncated = False
                if isclose(current_cost, best_cost):
                    if len(solutions) < self.equivalent_limit:
                        solutions.add(candidate)
                    elif candidate not in solutions:
                        truncated = True
                return
            uncovered_index = next(index for index in range(len(relevant)) if not mask & (1 << index))
            obligation = relevant[uncovered_index]
            candidates = sorted(
                (atom for atom in obligation.candidate_atoms if atom in masks),
                key=lambda atom: (
                    cost_model.cost(atom) / masks[atom].bit_count(),
                    cost_model.cost(atom),
                    atom.key,
                ),
            )
            for atom in candidates:
                if atom in chosen:
                    search(mask | masks[atom], chosen, current_cost)
                else:
                    search(mask | masks[atom], (*chosen, atom), current_cost + cost_model.cost(atom))

        search(0, (), 0.0)
        if not solutions:
            raise ValueError("no exact cut covers the repairable obligations")
        ordered = tuple(
            sorted(
                solutions,
                key=lambda cut: (len(cut), tuple(atom.key for atom in cut)),
            )
        )
        return OptimalCutSet(
            cuts=ordered,
            optimal_cost=float(best_cost),
            enumeration_complete=not truncated,
            truncated=truncated,
            lower_bound_count=len(ordered),
            optimality_proven=True,
        )

    @staticmethod
    def _remove_dominated(
        obligations: tuple[ValidationObligation, ...],
        atoms: tuple[DefectAtom, ...],
        costs: RepairCostModel,
    ) -> tuple[DefectAtom, ...]:
        coverage = {
            atom: frozenset(
                item.obligation_id
                for item in obligations
                if atom in item.candidate_atoms
            )
            for atom in atoms
        }
        retained = []
        for atom in atoms:
            dominated = any(
                other != atom
                and coverage[atom] < coverage[other]
                and costs.cost(atom) >= costs.cost(other)
                for other in atoms
            )
            if not dominated:
                retained.append(atom)
        return tuple(sorted(retained, key=lambda atom: atom.key))

    @staticmethod
    def _obligation_components(
        obligations: tuple[ValidationObligation, ...],
    ) -> tuple[tuple[ValidationObligation, ...], ...]:
        remaining = set(obligations)
        components = []
        while remaining:
            seed = min(remaining, key=lambda item: item.obligation_id)
            component = {seed}
            frontier = [seed]
            remaining.remove(seed)
            while frontier:
                current = frontier.pop()
                linked = [
                    item
                    for item in remaining
                    if set(current.candidate_atoms).intersection(item.candidate_atoms)
                ]
                for item in linked:
                    remaining.remove(item)
                    component.add(item)
                    frontier.append(item)
            components.append(tuple(sorted(component, key=lambda item: item.obligation_id)))
        return tuple(components)
