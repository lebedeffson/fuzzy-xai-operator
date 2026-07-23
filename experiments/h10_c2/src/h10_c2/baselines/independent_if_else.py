from __future__ import annotations

from time import perf_counter

from ..models import Case, MethodResult


def run(case: Case) -> MethodResult:
    started = perf_counter()
    cut = []
    actions = []
    for node in case.observed_route["nodes"]:
        registered = node["registered_attributes"]
        observed = node["observed_attributes"]
        mismatch = next((key for key in ("available", "sha256", "version", "schema") if registered.get(key) != observed.get(key)), None)
        if mismatch is None:
            continue
        atom = f"node:{node['node_id']}"
        cut.append(atom)
        actions.append(
            {
                "operation": "restore_from_registered_provider",
                "target_kind": "node",
                "target": node["node_id"],
                "field": "observed_attributes",
                "provider_ref": f"registry://{node['node_id']}",
            }
        )
    predicted = tuple(sorted(cut))
    return MethodResult(
        case.case_id,
        case.pipeline,
        "independent_if_else",
        predicted,
        sum(case.repair_costs.get(atom, 1.0) for atom in predicted),
        tuple(actions),
        runtime_ms=(perf_counter() - started) * 1000,
    )

