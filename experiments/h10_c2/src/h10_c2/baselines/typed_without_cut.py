from __future__ import annotations

from time import perf_counter

from ..models import Case, MethodResult


def run(case: Case) -> MethodResult:
    started = perf_counter()
    ordered = []
    for node in case.observed_route["nodes"]:
        if node["registered_attributes"] != node["observed_attributes"]:
            ordered.append(f"node:{node['node_id']}")
    cut = tuple(dict.fromkeys(ordered))
    actions = tuple({"operation": "request_typed_review", "target": atom.split(":", 1)[1]} for atom in cut)
    return MethodResult(case.case_id, case.pipeline, "typed_without_cut", cut, sum(case.repair_costs.get(atom, 1.0) for atom in cut), actions, runtime_ms=(perf_counter() - started) * 1000)

