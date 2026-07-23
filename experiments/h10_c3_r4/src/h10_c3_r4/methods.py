from __future__ import annotations

from dataclasses import replace
from decimal import Decimal
from time import perf_counter

from fuzzyxai.diagnostics import (
    DiagnosticValidator,
    MinimalDiagnosticCutFinder,
    RepairCostModel,
)

from .models import PublicCandidate, R4MethodResult

BASELINES = (
    "schema_only",
    "simple_or",
    "independent_if_else",
    "untyped_graph",
    "typed_without_optimization",
    "greedy_cover",
    "weighted_greedy",
)


def public_candidates(
    graph: object,
) -> tuple[PublicCandidate, ...]:
    validation = DiagnosticValidator().validate(graph)
    costs = RepairCostModel(graph.metadata.get("repair_costs", {}))
    coverage: dict[object, set[str]] = {}
    for obligation in validation.obligations:
        for atom in obligation.candidate_atoms:
            coverage.setdefault(atom, set()).add(obligation.obligation_id)
    return tuple(
        PublicCandidate(
            candidate_id=atom.key,
            subject_kind=atom.subject_kind,
            subject_id=atom.subject_id,
            field=atom.field,
            violation_code=atom.violation_code,
            covers=tuple(sorted(obligations)),
            cost=costs.cost(atom),
        )
        for atom, obligations in sorted(
            coverage.items(),
            key=lambda item: item[0].key,
        )
    )


def run_full_h10(graph: object) -> R4MethodResult:
    started = perf_counter()
    validation = DiagnosticValidator().validate(graph)
    actionable = replace(
        validation,
        obligations=tuple(
            replace(
                obligation,
                candidate_atoms=tuple(
                    atom
                    for atom in obligation.candidate_atoms
                    if atom.violation_code == "source_component"
                ),
            )
            for obligation in validation.obligations
        ),
    )
    cut = MinimalDiagnosticCutFinder().find(
        graph,
        actionable,
        RepairCostModel(graph.metadata.get("repair_costs", {})),
    )
    return R4MethodResult(
        method="full_h10",
        cut=tuple(sorted(cut.atom_keys)),
        predicted_cost=cut.total_cost,
        status="diagnosed" if not cut.uncovered_obligations else "insufficient_evidence",
        runtime_ms=(perf_counter() - started) * 1000,
    )


def _greedy(
    candidates: tuple[PublicCandidate, ...],
    obligations: set[str],
    *,
    weighted: bool,
) -> tuple[PublicCandidate, ...]:
    remaining = set(obligations)
    selected = []
    while remaining:
        useful = [
            candidate
            for candidate in candidates
            if remaining.intersection(candidate.covers)
        ]
        if not useful:
            break
        chosen = min(
            useful,
            key=lambda item: (
                (
                    Decimal(str(item.cost))
                    / len(remaining.intersection(item.covers))
                )
                if weighted
                else -len(remaining.intersection(item.covers)),
                Decimal(str(item.cost)),
                item.candidate_id,
            ),
        )
        selected.append(chosen)
        remaining.difference_update(chosen.covers)
    return tuple(selected)


def _baseline_cut(
    name: str,
    candidates: tuple[PublicCandidate, ...],
    obligations: tuple[str, ...],
) -> tuple[PublicCandidate, ...]:
    if name == "schema_only":
        return tuple(
            item
            for item in candidates
            if item.field and "schema" in item.field
        )[:1]
    if name == "simple_or":
        return candidates
    if name in {"untyped_graph", "greedy_cover"}:
        return _greedy(candidates, set(obligations), weighted=False)
    if name == "weighted_greedy":
        return _greedy(candidates, set(obligations), weighted=True)
    selected = []
    for obligation in obligations:
        options = sorted(
            (
                item
                for item in candidates
                if obligation in item.covers
            ),
            key=lambda item: (
                Decimal(str(item.cost)),
                item.candidate_id,
            ),
        )
        if options:
            selected.append(options[0])
    return tuple(dict.fromkeys(selected))


def run_baseline(name: str, graph: object) -> R4MethodResult:
    if name not in BASELINES:
        raise ValueError(f"unknown R4 baseline: {name}")
    started = perf_counter()
    candidates = public_candidates(graph)
    obligations = tuple(
        item.obligation_id
        for item in DiagnosticValidator().validate(graph).obligations
    )
    cut = _baseline_cut(name, candidates, obligations)
    covered = {
        obligation
        for candidate in cut
        for obligation in candidate.covers
    }
    return R4MethodResult(
        method=name,
        cut=tuple(sorted(item.candidate_id for item in cut)),
        predicted_cost=sum(item.cost for item in cut),
        status=(
            "diagnosed"
            if set(obligations).issubset(covered)
            else "insufficient_evidence"
        ),
        runtime_ms=(perf_counter() - started) * 1000,
    )
