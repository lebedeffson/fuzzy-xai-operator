from __future__ import annotations

from time import perf_counter

from ..models import Case, MethodResult


def run(case: Case) -> MethodResult:
    started = perf_counter()
    required = {"node_id", "node_type", "registered_attributes", "observed_attributes"}
    invalid = [node["node_id"] for node in case.observed_route["nodes"] if not required.issubset(node)]
    cut = tuple(f"node:{node}" for node in invalid)
    return MethodResult(case.case_id, case.pipeline, "schema_only", cut, float(len(cut)), runtime_ms=(perf_counter() - started) * 1000)

