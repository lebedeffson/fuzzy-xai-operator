from __future__ import annotations

from .contracts import ApproximateCut, DefectAtom, RepairCostModel, RouteGraph, ValidationObligation
from .cut_verification import covered_obligations, verify_cut_elements


class ApproximateMinimalCutSolver:
    def solve(
        self,
        graph: RouteGraph,
        obligations: tuple[ValidationObligation, ...],
        atoms: tuple[DefectAtom, ...],
        costs: RepairCostModel | None = None,
    ) -> ApproximateCut:
        cost_model = costs or RepairCostModel()
        if not verify_cut_elements(graph, atoms):
            raise ValueError("approximate solver received an atom outside the route graph")
        relevant = tuple(item for item in obligations if item.repairable and item.candidate_atoms)
        target = {item.obligation_id for item in relevant}
        starts = (
            lambda atom, gain: (cost_model.cost(atom) / gain, cost_model.cost(atom), atom.key),
            lambda atom, gain: (-gain, cost_model.cost(atom), atom.key),
            lambda atom, gain: (cost_model.cost(atom), -gain, atom.key),
        )
        candidates = []
        for ordering in starts:
            remaining = set(target)
            chosen: list[DefectAtom] = []
            while remaining:
                gains = {
                    atom: len(
                        remaining
                        & {
                            item.obligation_id
                            for item in relevant
                            if atom in item.candidate_atoms
                        }
                    )
                    for atom in atoms
                }
                useful = [atom for atom, gain in gains.items() if gain]
                if not useful:
                    break
                selected = min(useful, key=lambda atom: ordering(atom, gains[atom]))
                chosen.append(selected)
                remaining -= set(covered_obligations((selected,), relevant))
            candidates.append(self._prune(tuple(chosen), relevant))
        feasible = [cut for cut in candidates if target.issubset(covered_obligations(cut, relevant))]
        if not feasible:
            raise ValueError("approximate solver cannot cover the repairable obligations")
        selected = min(
            feasible,
            key=lambda cut: (
                sum(cost_model.cost(atom) for atom in cut),
                len(cut),
                tuple(atom.key for atom in cut),
            ),
        )
        cost = sum(cost_model.cost(atom) for atom in selected)
        lower_bound = max(
            (
                min(cost_model.cost(atom) for atom in item.candidate_atoms)
                for item in relevant
            ),
            default=0.0,
        )
        gap = (cost - lower_bound) / max(lower_bound, 1e-12)
        return ApproximateCut(selected, cost, lower_bound, gap, False)

    @staticmethod
    def _prune(
        cut: tuple[DefectAtom, ...],
        obligations: tuple[ValidationObligation, ...],
    ) -> tuple[DefectAtom, ...]:
        target = {item.obligation_id for item in obligations}
        selected = list(dict.fromkeys(cut))
        for atom in reversed(selected.copy()):
            candidate = tuple(item for item in selected if item != atom)
            if target.issubset(covered_obligations(candidate, obligations)):
                selected.remove(atom)
        return tuple(sorted(selected, key=lambda atom: atom.key))
