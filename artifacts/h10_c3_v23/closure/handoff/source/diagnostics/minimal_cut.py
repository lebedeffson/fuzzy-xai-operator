from __future__ import annotations

from time import perf_counter

from .approximate_solver import ApproximateMinimalCutSolver
from .contracts import (
    DefectAtom,
    DiagnosticCut,
    RepairCostModel,
    RouteGraph,
    ValidationObligation,
    ValidationResult,
)
from .cut_verification import covered_obligations, verify_cut
from .exact_solver import ExactMinimalCutSolver


class MinimalDiagnosticCutFinder:
    def __init__(
        self,
        *,
        exact_atom_limit: int = 32,
        exact_complexity_limit: int = 4096,
        equivalent_limit: int = 512,
    ) -> None:
        self.exact_atom_limit = exact_atom_limit
        self.exact_complexity_limit = exact_complexity_limit
        self.exact_solver = ExactMinimalCutSolver(equivalent_limit=equivalent_limit)
        self.approximate_solver = ApproximateMinimalCutSolver()

    def find(
        self,
        graph: RouteGraph,
        validation: ValidationResult,
        costs: RepairCostModel | None = None,
    ) -> DiagnosticCut:
        started = perf_counter()
        cost_model = costs or RepairCostModel()
        repairable = tuple(
            item
            for item in validation.obligations
            if item.repairable and item.candidate_atoms
        )
        atoms = tuple(
            sorted(
                {atom for item in repairable for atom in item.candidate_atoms},
                key=lambda atom: atom.key,
            )
        )
        if not validation.obligations:
            return DiagnosticCut(
                (),
                (),
                0.0,
                True,
                "none",
                (),
                (),
                ((),),
                (perf_counter() - started) * 1000,
                True,
                False,
                1,
                True,
                0.0,
                0.0,
            )
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
        complexity = self._complexity(repairable, atoms)
        if len(atoms) <= self.exact_atom_limit and complexity <= self.exact_complexity_limit:
            solution = self.exact_solver.solve(graph, repairable, atoms, cost_model)
            selected = solution.cuts[0]
            equivalent = solution.cuts
            total_cost = solution.optimal_cost
            solver = "exact_branch_and_bound"
            optimal = solution.optimality_proven
            enumeration_complete = solution.enumeration_complete
            truncated = solution.truncated
            lower_bound_count = solution.lower_bound_count
            optimality_proven = solution.optimality_proven
            lower_bound = total_cost
            optimality_gap = 0.0
        else:
            solution = self.approximate_solver.solve(graph, repairable, atoms, cost_model)
            selected = solution.cut
            equivalent = ()
            total_cost = solution.cost
            solver = "approximate_multistart_weighted_cover"
            optimal = False
            enumeration_complete = False
            truncated = False
            lower_bound_count = 0
            optimality_proven = False
            lower_bound = solution.lower_bound
            optimality_gap = solution.optimality_gap
        covered = covered_obligations(selected, repairable)
        uncovered = tuple(
            sorted(
                item.obligation_id
                for item in validation.obligations
                if item.obligation_id not in covered
            )
        )
        cut = DiagnosticCut(
            defect_atoms=selected,
            affected_nodes=tuple(
                sorted(
                    {
                        atom.subject_id
                        for atom in selected
                        if atom.subject_kind == "node"
                    }
                )
            ),
            total_cost=float(total_cost),
            optimal=optimal and not uncovered,
            solver=solver,
            covered_obligations=tuple(sorted(covered)),
            uncovered_obligations=uncovered,
            equivalent_optimal_cuts=equivalent,
            runtime_ms=(perf_counter() - started) * 1000,
            enumeration_complete=enumeration_complete,
            truncated=truncated,
            lower_bound_count=lower_bound_count,
            optimality_proven=optimality_proven,
            lower_bound=lower_bound,
            optimality_gap=optimality_gap,
        )
        verify_cut(graph, cut, validation.obligations, cost_model)
        return cut

    @staticmethod
    def _complexity(
        obligations: tuple[ValidationObligation, ...],
        atoms: tuple[DefectAtom, ...],
    ) -> int:
        branching = 1
        for obligation in obligations:
            branching *= max(1, len(obligation.candidate_atoms))
            if branching > 1_000_000:
                break
        density = sum(len(item.candidate_atoms) for item in obligations)
        return min(branching, 1_000_000) + density * max(1, len(atoms))


__all__ = [
    "ApproximateMinimalCutSolver",
    "ExactMinimalCutSolver",
    "MinimalDiagnosticCutFinder",
    "verify_cut",
]
