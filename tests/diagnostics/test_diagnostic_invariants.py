from __future__ import annotations

import inspect
from dataclasses import replace

import pytest

import fuzzyxai.diagnostics as diagnostics
from fuzzyxai.diagnostics import Contract, DiagnosticValidator, RouteGraphBuilder


def test_production_diagnostics_do_not_import_gold_oracle() -> None:
    sources = "\n".join(
        inspect.getsource(value)
        for value in (
            diagnostics.DiagnosticService,
            diagnostics.RepairExecutor,
            diagnostics.ActionableRepairPlanner,
            diagnostics.RouteRecertifier,
        )
    )
    forbidden = ("gold_oracle", "source_truth", "repair_truth", "mutation_log")
    assert not any(token in sources for token in forbidden)


def test_production_executor_does_not_copy_expected_to_observed() -> None:
    source = inspect.getsource(diagnostics.RepairExecutor)
    assert "expected" not in source
    assert "observed" not in source


def test_unknown_contract_is_insufficient_not_invented(valid_route: dict) -> None:
    route = {
        **valid_route,
        "contracts": [
            {
                "contract_id": "unknown:c1",
                "kind": "future_contract",
                "subject_id": "model",
                "repairable": False,
            }
        ],
    }
    graph = RouteGraphBuilder().build(route)
    result = DiagnosticValidator().validate(graph)
    assert result.status == "unknown"
    assert result.issues[0].unknown
    assert result.issues[0].cause_candidates[-1].status == "insufficient_evidence"


def test_missing_value_is_not_zero(valid_route: dict) -> None:
    route = {
        **valid_route,
        "contracts": [
            {
                "contract_id": "model:max_loss",
                "kind": "max_value",
                "subject_id": "model",
                "field": "loss",
                "expected": 0.1,
            }
        ],
    }
    result = DiagnosticValidator().validate(RouteGraphBuilder().build(route))
    assert result.status == "insufficient_evidence"
    assert result.issues[0].confidence is None


def test_cycle_is_reported_separately(valid_graph) -> None:
    reverse = replace(
        valid_graph.edges[0],
        edge_id="model-to-preprocessor",
        source="model",
        target="preprocessor",
    )
    graph = replace(valid_graph, edges=(*valid_graph.edges, reverse))
    result = DiagnosticValidator().validate(graph)
    assert any(issue.code == "cyclic_mandatory_route" for issue in result.issues)


def test_irreparable_issue_leaves_uncovered_obligation(valid_route: dict) -> None:
    route = {
        **valid_route,
        "contracts": [
            Contract(
                contract_id="irreparable",
                kind="equals",
                subject_id="model",
                field="version",
                expected="v9",
                repairable=False,
            ).__dict__
        ],
    }
    report = diagnostics.DiagnosticService().diagnose(route=route)
    assert report.minimal_cut is not None
    assert report.minimal_cut.uncovered_obligations == ("irreparable",)
    assert report.repair_plan is not None
    assert not report.repair_plan.fully_executable


def test_invalid_repair_mode_is_rejected(valid_route: dict) -> None:
    with pytest.raises(ValueError, match="repair_mode"):
        diagnostics.DiagnosticService().diagnose(route=valid_route, repair_mode="magic")
