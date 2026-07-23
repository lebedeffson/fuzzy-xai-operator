from __future__ import annotations

from time import perf_counter

from ..models import Case, MethodResult


def run(case: Case) -> MethodResult:
    started = perf_counter()
    remaining = [
        {f"node:{node['node_id']}"}
        for node in case.observed_route["nodes"]
        if node["registered_attributes"] != node["observed_attributes"]
    ]
    chosen = []
    while remaining:
        atoms = set().union(*remaining)
        if not atoms:
            break
        selected = max(atoms, key=lambda atom: (sum(atom in path for path in remaining), atom))
        chosen.append(selected)
        remaining = [path for path in remaining if selected not in path]
    cut = tuple(sorted(chosen))
    return MethodResult(case.case_id, case.pipeline, "greedy_cover", cut, sum(case.repair_costs.get(atom, 1.0) for atom in cut), optimality_claimed=False, runtime_ms=(perf_counter() - started) * 1000)
