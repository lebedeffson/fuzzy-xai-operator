from __future__ import annotations

from time import perf_counter

from fuzzyxai.diagnostics import (
    DefectAtom,
    ExactMinimalCutSolver,
    RepairCostModel,
    RouteEdge,
    RouteGraph,
    RouteNode,
    ValidationObligation,
)

from .models import Candidate, MethodResult


def _candidates(view: dict[str, object]) -> tuple[Candidate, ...]:
    return tuple(
        Candidate(
            **{
                **value,
                "covers": tuple(value["covers"]),
                "dependencies": tuple(value["dependencies"]),
                "alternatives": tuple(value["alternatives"]),
                "conflicts": tuple(value["conflicts"]),
            }
        )
        for value in view["candidates"]
    )


def _route(view: dict[str, object], candidates: tuple[Candidate, ...]) -> RouteGraph:
    nodes = tuple(
        RouteNode(
            node_id=node_id,
            node_type="artifact",
            component_id=node_id,
            component_version="v23",
            registered_attributes={},
            observed_attributes={},
            mandatory=True,
            repairable=True,
            evidence_refs=(f"evidence:{node_id}",),
        )
        for node_id in view["nodes"]
    )
    edges = tuple(
        RouteEdge(
            edge_id=edge_id,
            source=source,
            target=target,
            relation="transforms",
            mandatory=True,
            registered_contract={"relation": "transforms"},
            observed_contract={"relation": "transforms"},
            repairable=True,
            relation_status="known_valid",
        )
        for edge_id, source, target in view["edges"]
    )
    return RouteGraph(
        route_id=str(view["case_id"]),
        nodes=nodes,
        edges=edges,
        contracts=(),
        metadata={"candidate_count": len(candidates)},
    )


def run_fuzzyxai(view: dict[str, object]) -> MethodResult:
    started = perf_counter()
    candidates = _candidates(view)
    usable = tuple(
        candidate
        for candidate in candidates
        if candidate.repairable
        and candidate.executable
        and candidate.provider_status == "healthy"
        and candidate.covers
    )
    atoms = tuple(
        DefectAtom(
            subject_kind=candidate.subject_kind,
            subject_id=candidate.subject_id,
            field=candidate.field,
            violation_code=candidate.violation_code,
            repairable=candidate.repairable,
            repair_cost=candidate.cost,
        )
        for candidate in usable
    )
    atom_by_id = dict(zip((item.atom_id for item in usable), atoms, strict=True))
    id_by_atom = {atom: atom_id for atom_id, atom in atom_by_id.items()}
    obligations = tuple(
        ValidationObligation(
            obligation_id=obligation,
            issue_id=f"issue:{obligation}",
            candidate_atoms=tuple(
                atom_by_id[item.atom_id] for item in usable if obligation in item.covers
            ),
            repairable=True,
        )
        for obligation in view["obligations"]
    )
    if any(not item.candidate_atoms for item in obligations):
        return MethodResult(
            "full_fuzzyxai",
            (),
            (),
            0.0,
            (perf_counter() - started) * 1000,
            "insufficient_evidence",
        )
    result = ExactMinimalCutSolver().solve(
        _route(view, usable),
        obligations,
        atoms,
        RepairCostModel(),
    )
    cut_ids = tuple(sorted(id_by_atom[atom] for atom in result.cuts[0]))
    all_candidates = {item.atom_id: item for item in candidates}
    plan: list[str] = []

    def add_with_dependencies(atom_id: str) -> None:
        item = all_candidates[atom_id]
        for dependency in item.dependencies:
            if dependency in all_candidates and dependency not in plan:
                add_with_dependencies(dependency)
        if atom_id not in plan:
            plan.append(atom_id)

    for atom_id in cut_ids:
        add_with_dependencies(atom_id)
    return MethodResult(
        method="full_fuzzyxai",
        cut=cut_ids,
        plan=tuple(plan),
        predicted_cost=result.optimal_cost,
        runtime_ms=(perf_counter() - started) * 1000,
        status="diagnosed",
    )

