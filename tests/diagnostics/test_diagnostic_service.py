from __future__ import annotations

from dataclasses import replace

import pytest

from fuzzyxai import FuzzyXAI
from fuzzyxai.diagnostics import RepairExecutionContext, RouteGraph


def test_public_api_diagnoses_valid_route(valid_route: dict) -> None:
    report = FuzzyXAI().diagnose(route=valid_route)
    assert report.route_status == "valid"
    assert report.issues == ()
    assert report.minimal_cut is None
    assert "не доказывает ошибочность прогноза" in report.limitations[0]


def test_public_api_builds_evidence_bound_plan(invalid_route: dict) -> None:
    report = FuzzyXAI().diagnose(route=invalid_route, repair_mode="plan")
    assert report.route_status == "invalid"
    assert len(report.issues) == 2
    assert all(issue.evidence_refs for issue in report.issues)
    assert report.minimal_cut is not None
    assert report.minimal_cut.uncovered_obligations == ()
    assert report.repair_plan is not None
    assert report.repair_plan.steps
    assert all("expected" not in step.parameters for step in report.repair_plan.steps)
    assert report.repair_plan.fully_executable is False


def test_execute_is_fail_closed_without_context(invalid_route: dict) -> None:
    with pytest.raises(PermissionError, match="explicit RepairExecutionContext"):
        FuzzyXAI().diagnose(route=invalid_route, repair_mode="execute")


def test_execute_requires_external_permission(invalid_route: dict) -> None:
    context = RepairExecutionContext(handlers={})
    with pytest.raises(PermissionError, match="allow_external_changes"):
        FuzzyXAI().diagnose(route=invalid_route, repair_mode="execute", repair_context=context)


def test_explicit_handler_is_recertified(invalid_route: dict) -> None:
    planned = FuzzyXAI().diagnose(route=invalid_route)
    assert planned.repair_plan

    def restore(graph: RouteGraph, step) -> RouteGraph:
        nodes = []
        for node in graph.nodes:
            if node.node_id == step.target.subject_id:
                nodes.append(replace(node, observed_attributes=dict(node.registered_attributes)))
            else:
                nodes.append(node)
        return replace(graph, nodes=tuple(nodes))

    operations = {step.operation: restore for step in planned.repair_plan.steps}
    approved = frozenset(step.step_id for step in planned.repair_plan.steps)
    context = RepairExecutionContext(operations, approved, allow_external_changes=True)
    report = FuzzyXAI().diagnose(route=invalid_route, repair_mode="execute", repair_context=context)
    assert report.recertification is not None
    assert report.recertification.route_valid_after
    assert report.recertification.status == "full_success"


def test_reports_are_byte_deterministic(invalid_route: dict) -> None:
    first = FuzzyXAI().diagnose(route=invalid_route)
    second = FuzzyXAI().diagnose(route=invalid_route)
    assert first.trace == second.trace
    assert first.trace_sha256 == second.trace_sha256


def test_batch_returns_aggregate_counts(valid_route: dict, invalid_route: dict) -> None:
    batch = FuzzyXAI().diagnose_batch(routes=(valid_route, invalid_route))
    assert len(batch.reports) == 2
    assert batch.route_status_counts == {"invalid": 1, "valid": 1}
    assert batch.issue_category_counts["preprocessing"] == 2
