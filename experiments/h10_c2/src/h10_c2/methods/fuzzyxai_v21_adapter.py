from __future__ import annotations

from time import perf_counter

from fuzzyxai import FuzzyXAI

from ..models import Case, MethodResult


def run(case: Case) -> MethodResult:
    started = perf_counter()
    report = FuzzyXAI().diagnose(route=case.observed_route, repair_mode="plan")
    cut = ()
    optimality = False
    if report.minimal_cut is not None:
        cut = tuple(f"node:{node}" for node in report.minimal_cut.affected_nodes)
        optimality = bool(report.minimal_cut.optimal)
    actions = ()
    if report.repair_plan is not None:
        actions = tuple(
            {
                "operation": step.operation,
                "target": step.target,
                "provider_ref": step.parameters.get("registered_source_ref"),
                "preconditions": list(step.preconditions),
            }
            for step in report.repair_plan.steps
        )
    return MethodResult(
        case.case_id,
        case.pipeline,
        "fuzzyxai_v21",
        tuple(sorted(set(cut))),
        float(report.minimal_cut.total_cost if report.minimal_cut else 0.0),
        actions,
        optimality,
        runtime_ms=(perf_counter() - started) * 1000,
    )

