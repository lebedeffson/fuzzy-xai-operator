from __future__ import annotations

from collections import Counter
from time import perf_counter

from ..models import Case, MethodResult


def run(case: Case) -> MethodResult:
    started = perf_counter()
    suspicious = {
        node["node_id"]
        for node in case.observed_route["nodes"]
        if node.get("registered_attributes") != node.get("observed_attributes")
    }
    degree = Counter()
    for edge in case.observed_route["edges"]:
        degree[edge["source"]] += 1
        degree[edge["target"]] += 1
    if len(suspicious) > 1:
        common = max(suspicious, key=lambda node: (degree[node], node))
        cut = (f"node:{common}",)
    else:
        cut = tuple(f"node:{node}" for node in sorted(suspicious))
    return MethodResult(case.case_id, case.pipeline, "untyped_graph", cut, sum(case.repair_costs.get(atom, 1.0) for atom in cut), runtime_ms=(perf_counter() - started) * 1000)

