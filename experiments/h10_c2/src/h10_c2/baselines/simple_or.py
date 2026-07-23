from __future__ import annotations

from time import perf_counter

from ..models import Case, MethodResult


def run(case: Case) -> MethodResult:
    started = perf_counter()
    flagged = []
    for node in case.observed_route["nodes"]:
        if any(node["registered_attributes"].get(key) != node["observed_attributes"].get(key) for key in node["registered_attributes"]):
            flagged.append(f"node:{node['node_id']}")
    cut = tuple(sorted(flagged))
    actions = tuple({"operation": "request_manual_restore", "target": atom.split(":", 1)[1]} for atom in cut)
    cost = sum(case.repair_costs.get(atom, 1.0) for atom in cut)
    return MethodResult(case.case_id, case.pipeline, "simple_or", cut, cost, actions, runtime_ms=(perf_counter() - started) * 1000)

