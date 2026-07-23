from __future__ import annotations

from dataclasses import replace

import pytest

from fuzzyxai.audit_h10.models import RouteObservation
from fuzzyxai.core.types import OperatorEdge, OperatorNode, OperatorRoute, ProofTrace
from fuzzyxai.diagnostics import (
    ContractRegistry,
    DiagnosticService,
    DiagnosticValidator,
    RepairExecutionContext,
    RepairExecutor,
    RepairProviderRegistry,
    RouteGraphBuilder,
    RouteRecertifier,
)
from fuzzyxai.diagnostics.contract_registry import ContractCheck
from fuzzyxai.diagnostics.contracts import StepExecutionResult


def test_builder_supports_operator_route_and_proof_trace() -> None:
    node = OperatorNode("n1", "Node", "in", "out", "value", raw={"version": "v1"})
    edge = OperatorEdge("e1", "n1", "n1", {"relation": "validates"}, "self-check")
    route = OperatorRoute("scenario", "Scenario", [node], {}, [], "review", edges=[edge])
    graph = RouteGraphBuilder().build(route)
    assert graph.metadata["source_type"] == "OperatorRoute"
    proof = ProofTrace("FuzzyXAIProofTrace", "1.0", "scenario", route.to_dict(), {}, [], "review")
    assert RouteGraphBuilder().build(proof).metadata["proof_schema_version"] == "1.0"


def test_builder_supports_old_observation_and_nested_mapping() -> None:
    observation = RouteObservation(
        "old",
        "dataset",
        "tabular",
        "object",
        {"version": "v1"},
        {"version": "v2"},
        ("version",),
        (("version",),),
        {"version": 2.0},
    )
    graph = RouteGraphBuilder().build(observation)
    assert graph.metadata["source_type"] == "RouteObservation"
    assert graph.metadata["repair_costs"] == {"version": 2.0}
    nested = RouteGraphBuilder().build({"route": graph.to_dict()})
    assert nested.route_id == "old"
    with pytest.raises(TypeError, match="unsupported"):
        RouteGraphBuilder().build(object())


@pytest.mark.parametrize(
    ("kind", "field", "expected", "parameters", "actual", "status"),
    (
        ("max_value", "loss", 0.1, {}, 0.2, "invalid"),
        ("min_value", "quality", 0.9, {}, 0.5, "invalid"),
        ("allowed", "state", None, {"allowed": ["ready"]}, "blocked", "invalid"),
        ("checksum", "sha256", "abc", {}, "def", "invalid"),
        ("required_attribute", "tokenizer", None, {}, None, "insufficient_evidence"),
    ),
)
def test_contract_evaluators(
    valid_route: dict,
    kind: str,
    field: str,
    expected: object,
    parameters: dict,
    actual: object,
    status: str,
) -> None:
    route = {
        **valid_route,
        "nodes": [dict(item) for item in valid_route["nodes"]],
        "contracts": [
            {
                "contract_id": f"c:{kind}",
                "kind": kind,
                "subject_id": "model",
                "field": field,
                "expected": expected,
                "parameters": parameters,
            }
        ],
    }
    route["nodes"][1]["observed_attributes"] = {
        **route["nodes"][1]["observed_attributes"],
        field: actual,
    }
    assert DiagnosticValidator().validate(RouteGraphBuilder().build(route)).status == status


def test_custom_contract_registration(valid_graph) -> None:
    registry = ContractRegistry.default()
    registry.register("always", lambda contract, subject, graph: ContractCheck(True, False, True, "ok"))
    custom = replace(
        valid_graph,
        contracts=(
            replace(valid_graph.contracts[0], contract_id="custom", kind="always"),
        ),
    )
    assert DiagnosticValidator(registry).validate(custom).valid
    with pytest.raises(ValueError, match="cannot be empty"):
        registry.register("", lambda contract, subject, graph: ContractCheck(True, False, True, "ok"))


def test_executor_records_not_approved_missing_handler_and_failure(invalid_route: dict) -> None:
    service = DiagnosticService()
    graph = service.builder.build(invalid_route)
    validation = service.validator.validate(graph)
    cut = service.cut_finder.find(graph, validation)
    plan = service.repair_planner.plan(graph, validation.issues, cut)
    executor = RepairExecutor(service.repair_planner.registry)
    unchanged, results = executor.execute(
        graph,
        plan,
        RepairExecutionContext({}, frozenset(), allow_external_changes=True),
    )
    assert unchanged == graph
    assert {result.status for result in results} == {"not_approved"}
    approved = frozenset(step.step_id for step in plan.steps)
    _, results = executor.execute(
        graph,
        plan,
        RepairExecutionContext({}, approved, allow_external_changes=True),
    )
    assert {result.status for result in results} == {"handler_unavailable"}

    def broken_handler(graph, step):
        raise RuntimeError("external failure")

    _, results = executor.execute(
        graph,
        plan,
        RepairExecutionContext(
            {step.operation: broken_handler for step in plan.steps},
            approved,
            allow_external_changes=True,
        ),
    )
    assert {result.status for result in results} == {"failed"}


def test_recertifier_distinguishes_no_change_partial_and_degraded(invalid_route: dict) -> None:
    service = DiagnosticService()
    graph = service.builder.build(invalid_route)
    validation = service.validator.validate(graph)
    cut = service.cut_finder.find(graph, validation)
    plan = service.repair_planner.plan(graph, validation.issues, cut)
    no_change = RouteRecertifier().recertify(graph, graph, plan, ())
    assert no_change.status == "no_change"

    partially_fixed = replace(
        graph,
        nodes=tuple(
            replace(node, observed_attributes={**node.observed_attributes, "version": "v1"})
            if node.node_id == "preprocessor"
            else node
            for node in graph.nodes
        ),
    )
    partial = RouteRecertifier().recertify(
        graph,
        partially_fixed,
        plan,
        (StepExecutionResult(plan.steps[0].step_id, "completed", True),),
    )
    assert partial.status == "partial_success"

    new_contract = replace(graph.contracts[0], contract_id="new", kind="equals", field="missing", expected="x")
    degraded_graph = replace(graph, contracts=(*graph.contracts, new_contract))
    degraded = RouteRecertifier().recertify(graph, degraded_graph, plan, ())
    assert degraded.status == "worsened"


def test_repair_registry_rejects_duplicates_and_missing_provider() -> None:
    registry = RepairProviderRegistry()
    with pytest.raises(ValueError, match="already registered"):
        registry.register(registry.providers[0])
    with pytest.raises(KeyError):
        registry.get("missing")
