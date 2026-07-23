from __future__ import annotations

from time import perf_counter

from .models import Candidate, MethodResult

BASELINES = (
    "schema_only",
    "simple_or",
    "independent_if_else",
    "untyped_graph",
    "typed_without_optimization",
    "greedy_cover",
    "weighted_greedy",
)


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


def _greedy(
    candidates: tuple[Candidate, ...],
    obligations: set[str],
    *,
    weighted: bool,
) -> tuple[Candidate, ...]:
    remaining = set(obligations)
    selected: list[Candidate] = []
    while remaining:
        useful = [
            candidate
            for candidate in candidates
            if candidate.repairable and candidate.executable and remaining.intersection(candidate.covers)
        ]
        if not useful:
            break
        if weighted:
            chosen = min(
                useful,
                key=lambda item: (
                    item.cost / len(remaining.intersection(item.covers)),
                    item.cost,
                    item.atom_id,
                ),
            )
        else:
            chosen = min(
                useful,
                key=lambda item: (-len(remaining.intersection(item.covers)), item.atom_id),
            )
        selected.append(chosen)
        remaining.difference_update(chosen.covers)
    return tuple(selected)


def _cut(name: str, view: dict[str, object]) -> tuple[Candidate, ...]:
    candidates = _candidates(view)
    obligations = tuple(view["obligations"])
    if name == "schema_only":
        return tuple(item for item in candidates if item.field == "schema")[:1]
    if name == "simple_or":
        return tuple(item for item in candidates if item.covers)
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
                if obligation in item.covers and item.repairable and item.executable
            ),
            key=lambda item: (item.cost, item.atom_id),
        )
        if options:
            selected.append(options[0])
    return tuple(dict.fromkeys(selected))


def run_baseline(name: str, view: dict[str, object]) -> MethodResult:
    started = perf_counter()
    cut = _cut(name, view)
    plan = list(item.atom_id for item in cut)
    if name == "typed_without_optimization":
        by_id = {item.atom_id: item for item in _candidates(view)}
        dependencies = [
            dependency
            for item in cut
            for dependency in item.dependencies
            if dependency in by_id
        ]
        plan = [*dict.fromkeys(dependencies), *plan]
    return MethodResult(
        method=name,
        cut=tuple(sorted(item.atom_id for item in cut)),
        plan=tuple(plan),
        predicted_cost=sum(item.cost for item in cut),
        runtime_ms=(perf_counter() - started) * 1000,
        status="diagnosed" if cut else "insufficient_evidence",
    )


def with_cost_multiplier(view: dict[str, object], multiplier: float) -> dict[str, object]:
    adjusted = dict(view)
    adjusted["candidates"] = [
        {
            **value,
            "cost": value["cost"] * multiplier
            if value["atom_id"].startswith(("greedy-", "direct-"))
            else value["cost"],
        }
        for value in view["candidates"]
    ]
    return adjusted

