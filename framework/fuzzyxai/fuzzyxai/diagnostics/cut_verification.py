from __future__ import annotations

from dataclasses import dataclass
from math import isclose

from .contracts import (
    DefectAtom,
    DiagnosticCut,
    RepairCostModel,
    RouteGraph,
    ValidationObligation,
)


@dataclass(frozen=True)
class CutVerification:
    elements_valid: bool
    coverage_valid: bool
    cost_valid: bool
    feasibility_valid: bool
    optimality_valid: bool | None
    equivalent_optima_valid: bool | None

    @property
    def passed(self) -> bool:
        required = (
            self.elements_valid,
            self.coverage_valid,
            self.cost_valid,
            self.feasibility_valid,
        )
        optional = tuple(
            value
            for value in (self.optimality_valid, self.equivalent_optima_valid)
            if value is not None
        )
        return all((*required, *optional))


def _subject_exists(graph: RouteGraph, atom: DefectAtom) -> bool:
    if atom.subject_kind == "node":
        return graph.node(atom.subject_id) is not None or graph.node(f"node:{atom.subject_id}") is not None
    if atom.subject_kind == "edge":
        return graph.edge(atom.subject_id) is not None or graph.edge(f"edge:{atom.subject_id}") is not None
    return any(contract.contract_id == atom.subject_id for contract in graph.contracts)


def verify_cut_elements(graph: RouteGraph, atoms: tuple[DefectAtom, ...]) -> bool:
    return len(set(atoms)) == len(atoms) and all(_subject_exists(graph, atom) for atom in atoms)


def covered_obligations(
    atoms: tuple[DefectAtom, ...],
    obligations: tuple[ValidationObligation, ...],
) -> frozenset[str]:
    selected = set(atoms)
    return frozenset(
        obligation.obligation_id
        for obligation in obligations
        if selected.intersection(obligation.candidate_atoms)
    )


def verify_cut_coverage(
    cut: DiagnosticCut,
    obligations: tuple[ValidationObligation, ...],
) -> bool:
    covered = covered_obligations(cut.defect_atoms, obligations)
    expected = {item.obligation_id for item in obligations}
    return (
        covered == frozenset(cut.covered_obligations)
        and expected - covered == set(cut.uncovered_obligations)
    )


def verify_cut_cost(cut: DiagnosticCut, costs: RepairCostModel) -> bool:
    return isclose(
        sum(costs.cost(atom) for atom in cut.defect_atoms),
        cut.total_cost,
        rel_tol=1e-12,
        abs_tol=1e-12,
    )


def verify_cut_feasibility(
    graph: RouteGraph,
    cut: DiagnosticCut,
    obligations: tuple[ValidationObligation, ...],
) -> bool:
    if not verify_cut_elements(graph, cut.defect_atoms):
        return False
    if any(not atom.repairable for atom in cut.defect_atoms):
        return False
    repairable_ids = {
        item.obligation_id
        for item in obligations
        if item.repairable and item.candidate_atoms
    }
    return repairable_ids.issubset(cut.covered_obligations)


def _independent_optimum(
    obligations: tuple[ValidationObligation, ...],
    costs: RepairCostModel,
    *,
    solution_limit: int,
) -> tuple[float, tuple[tuple[DefectAtom, ...], ...], bool]:
    relevant = tuple(item for item in obligations if item.repairable and item.candidate_atoms)
    if not relevant:
        return 0.0, ((),), True
    index = {item.obligation_id: bit for bit, item in enumerate(relevant)}
    full_mask = (1 << len(relevant)) - 1
    atom_masks: dict[DefectAtom, int] = {}
    for obligation in relevant:
        bit = 1 << index[obligation.obligation_id]
        for atom in obligation.candidate_atoms:
            atom_masks[atom] = atom_masks.get(atom, 0) | bit
    states: dict[int, tuple[float, set[tuple[DefectAtom, ...]], bool]] = {0: (0.0, {()}, True)}
    for atom in sorted(atom_masks, key=lambda value: (costs.cost(value), value.key)):
        snapshot = tuple(states.items())
        for mask, (current_cost, current_cuts, complete) in snapshot:
            new_mask = mask | atom_masks[atom]
            new_cost = current_cost + costs.cost(atom)
            previous = states.get(new_mask)
            candidates = {
                tuple(sorted((*cut, atom), key=lambda value: value.key))
                for cut in current_cuts
            }
            if previous is None or new_cost < previous[0] and not isclose(new_cost, previous[0]):
                ordered_candidates = sorted(
                    candidates,
                    key=lambda value: (len(value), tuple(atom.key for atom in value)),
                )
                states[new_mask] = (
                    new_cost,
                    set(ordered_candidates[:solution_limit]),
                    complete,
                )
            elif isclose(new_cost, previous[0]):
                merged = previous[1] | candidates
                states[new_mask] = (
                    previous[0],
                    set(sorted(merged, key=lambda value: (len(value), tuple(a.key for a in value)))[:solution_limit]),
                    previous[2] and complete and len(merged) <= solution_limit,
                )
    if full_mask not in states:
        raise ValueError("no feasible cut covers all repairable obligations")
    best_cost, cuts, complete = states[full_mask]
    return (
        best_cost,
        tuple(sorted(cuts, key=lambda value: (len(value), tuple(atom.key for atom in value)))),
        complete,
    )


def verify_cut_optimality(
    cut: DiagnosticCut,
    obligations: tuple[ValidationObligation, ...],
    costs: RepairCostModel,
) -> bool:
    best_cost, _, _ = _independent_optimum(obligations, costs, solution_limit=1)
    return isclose(cut.total_cost, best_cost, rel_tol=1e-12, abs_tol=1e-12)


def verify_equivalent_optima(
    cut: DiagnosticCut,
    obligations: tuple[ValidationObligation, ...],
    costs: RepairCostModel,
) -> bool:
    if not cut.enumeration_complete:
        return not cut.optimality_proven or cut.truncated
    best_cost, independent, complete = _independent_optimum(
        obligations,
        costs,
        solution_limit=max(1, len(cut.equivalent_optimal_cuts) + 1),
    )
    return (
        complete
        and isclose(cut.total_cost, best_cost, rel_tol=1e-12, abs_tol=1e-12)
        and set(independent) == set(cut.equivalent_optimal_cuts)
    )


def verify_cut(
    graph: RouteGraph,
    cut: DiagnosticCut,
    obligations: tuple[ValidationObligation, ...],
    costs: RepairCostModel | None = None,
) -> CutVerification:
    cost_model = costs or RepairCostModel()
    result = CutVerification(
        elements_valid=verify_cut_elements(graph, cut.defect_atoms),
        coverage_valid=verify_cut_coverage(cut, obligations),
        cost_valid=verify_cut_cost(cut, cost_model),
        feasibility_valid=verify_cut_feasibility(graph, cut, obligations),
        optimality_valid=verify_cut_optimality(cut, obligations, cost_model) if cut.optimality_proven else None,
        equivalent_optima_valid=(
            verify_equivalent_optima(cut, obligations, cost_model)
            if cut.optimality_proven
            else None
        ),
    )
    if not result.passed:
        raise ValueError(f"diagnostic cut verification failed: {result}")
    return result
