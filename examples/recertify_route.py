from dataclasses import replace

from fuzzyxai import FuzzyXAI
from fuzzyxai.diagnostics import RepairExecutionContext, RouteGraph

from diagnose_single_route import route


planned = FuzzyXAI().diagnose(route=route)
assert planned.repair_plan


def external_registry_handler(graph: RouteGraph, step) -> RouteGraph:
    """Example only: a deployment integration resolves registry:// references."""
    nodes = tuple(
        replace(node, observed_attributes={"version": "v1"})
        if node.node_id == step.target
        else node
        for node in graph.nodes
    )
    return replace(graph, nodes=nodes)


context = RepairExecutionContext(
    handlers={step.operation: external_registry_handler for step in planned.repair_plan.steps},
    approved_step_ids=frozenset(step.step_id for step in planned.repair_plan.steps),
    allow_external_changes=True,
)
report = FuzzyXAI().diagnose(route=route, repair_mode="execute", repair_context=context)
print(report.recertification)
