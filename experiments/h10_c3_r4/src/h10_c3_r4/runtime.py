from __future__ import annotations

from dataclasses import replace

from fuzzyxai.diagnostics import (
    ActionableRepairPlanner,
    DiagnosticCut,
    DiagnosticValidator,
    RepairExecutionContext,
    RepairExecutor,
    RepairPlan,
    RepairProviderRegistry,
    RepairStep,
    RouteGraph,
    RouteRecertifier,
)

from .models import R4MethodResult


def _cut_from_result(
    graph: RouteGraph,
    result: R4MethodResult,
) -> tuple[DiagnosticCut, object]:
    validation = DiagnosticValidator().validate(graph)
    atoms = {
        atom.key: atom
        for obligation in validation.obligations
        for atom in obligation.candidate_atoms
    }
    selected = tuple(
        atoms[candidate_id]
        for candidate_id in result.cut
        if candidate_id in atoms
    )
    covered = {
        obligation.obligation_id
        for obligation in validation.obligations
        if any(atom in selected for atom in obligation.candidate_atoms)
    }
    all_obligations = {
        obligation.obligation_id for obligation in validation.obligations
    }
    cut = DiagnosticCut(
        defect_atoms=selected,
        affected_nodes=tuple(
            sorted(
                atom.subject_id
                for atom in selected
                if atom.subject_kind == "node"
            )
        ),
        total_cost=result.predicted_cost,
        optimal=result.method == "full_h10",
        solver=(
            "h10_exact"
            if result.method == "full_h10"
            else f"baseline:{result.method}"
        ),
        covered_obligations=tuple(sorted(covered)),
        uncovered_obligations=tuple(sorted(all_obligations - covered)),
        equivalent_optimal_cuts=(),
        runtime_ms=result.runtime_ms,
        optimality_proven=result.method == "full_h10",
    )
    return cut, validation


def _baseline_plan(
    graph: RouteGraph,
    result: R4MethodResult,
    cut: DiagnosticCut,
    validation: object,
    registry: RepairProviderRegistry,
) -> RepairPlan:
    selected = {(atom.subject_kind, atom.subject_id) for atom in cut.defect_atoms}
    steps = []
    for index, (kind, subject_id) in enumerate(sorted(selected), 1):
        relevant = tuple(
            issue
            for issue in validation.issues
            if subject_id in issue.source_nodes
            or subject_id in issue.affected_nodes
            or subject_id in issue.affected_edges
        )
        if not relevant:
            continue
        provider = registry.select(relevant[0])
        if provider is None:
            continue
        proposed = tuple(
            step
            for step in provider.propose(graph, relevant[0])
            if step.target.subject_kind == kind
            and step.target.subject_id == subject_id
        )
        if not proposed:
            continue
        base = proposed[0]
        contract_ids = tuple(
            issue.violated_contract
            for issue in relevant
        )
        steps.append(
            replace(
                base,
                step_id=f"baseline-step:{index}:{subject_id}",
                parameters={
                    **base.parameters,
                    "contract_ids": contract_ids,
                },
                expected_postconditions=tuple(
                    f"contract_satisfied:{contract_id}"
                    for contract_id in contract_ids
                ),
                depends_on=(),
            )
        )
    return RepairPlan(
        plan_id=f"baseline-plan:{graph.route_id}:{result.method}",
        cut=cut,
        steps=tuple(steps),
        total_estimated_cost=sum(
            step.estimated_cost or 0.0 for step in steps
        ),
        fully_executable=bool(steps) and not cut.uncovered_obligations,
        unresolved_issues=cut.uncovered_obligations,
        trace_sha256="baseline-independent-plan",
    )


def _restore_contracts(
    graph: RouteGraph,
    step: RepairStep,
) -> RouteGraph:
    dependencies = dict(graph.metadata.get("repair_dependencies", {}))
    repaired_sources = set(graph.metadata.get("repaired_sources", ()))
    missing = set(dependencies.get(step.target.subject_id, ())) - repaired_sources
    if missing:
        raise RuntimeError(
            f"repair dependency is not satisfied: {sorted(missing)}"
        )
    contract_ids = tuple(
        str(value)
        for value in (
            *step.parameters.get("contract_ids", ()),
            step.parameters.get("contract_id"),
        )
        if value
    )
    contracts = {
        contract.contract_id: contract for contract in graph.contracts
    }
    nodes = {node.node_id: node for node in graph.nodes}
    edges = {edge.edge_id: edge for edge in graph.edges}
    for contract_id in contract_ids:
        contract = contracts.get(contract_id)
        if contract is None:
            continue
        node = nodes.get(contract.subject_id)
        if node is not None:
            observed = dict(node.observed_attributes)
            observed[str(contract.field)] = contract.expected
            nodes[node.node_id] = replace(
                node,
                observed_attributes=observed,
            )
            continue
        edge = edges.get(contract.subject_id)
        if edge is not None:
            observed = dict(edge.observed_contract)
            observed[str(contract.field)] = contract.expected
            edges[edge.edge_id] = replace(
                edge,
                observed_contract=observed,
            )
    repaired_sources.add(step.target.subject_id)
    return replace(
        graph,
        nodes=tuple(nodes[node.node_id] for node in graph.nodes),
        edges=tuple(edges[edge.edge_id] for edge in graph.edges),
        metadata={
            **graph.metadata,
            "repaired_sources": tuple(sorted(repaired_sources)),
        },
    )


def execute_and_recertify(
    graph: RouteGraph,
    result: R4MethodResult,
) -> dict[str, object]:
    cut, validation = _cut_from_result(graph, result)
    registry = RepairProviderRegistry()
    if result.method == "full_h10":
        plan = ActionableRepairPlanner(registry).plan(
            graph,
            validation.issues,
            cut,
        )
    else:
        plan = _baseline_plan(
            graph,
            result,
            cut,
            validation,
            registry,
        )
    operations = {
        step.operation: _restore_contracts for step in plan.steps
    }
    context = RepairExecutionContext(
        handlers=operations,
        approved_step_ids=frozenset(step.step_id for step in plan.steps),
        allow_external_changes=True,
        satisfied_preconditions=frozenset(
            condition
            for step in plan.steps
            for condition in step.preconditions
        ),
    )
    repaired, execution = RepairExecutor(registry).execute(
        graph,
        plan,
        context,
    )
    recertification = RouteRecertifier().recertify(
        graph,
        repaired,
        plan,
        execution,
    )
    full_success = (
        recertification.status == "full_success"
        and recertification.remaining_critical_issues == ()
        and recertification.new_critical_issues == ()
        and recertification.all_required_postconditions_verified
    )
    return {
        "full_recertification_success": full_success,
        "recertification_status": recertification.status,
        "remaining_critical_issues": recertification.remaining_critical_issues,
        "new_critical_issues": recertification.new_critical_issues,
        "remaining_critical_issue_count": len(
            recertification.remaining_critical_issues
        ),
        "new_critical_violation_count": len(
            recertification.new_critical_issues
        ),
        "all_required_postconditions_verified": (
            recertification.all_required_postconditions_verified
        ),
        "completed_steps": len(recertification.completed_steps),
        "failed_steps": len(recertification.failed_steps),
        "plan_cost": plan.total_estimated_cost,
        "before_trace_sha256": recertification.before_trace_sha256,
        "after_trace_sha256": recertification.after_trace_sha256,
        "graph_changed": (
            recertification.before_trace_sha256
            != recertification.after_trace_sha256
        ),
    }
