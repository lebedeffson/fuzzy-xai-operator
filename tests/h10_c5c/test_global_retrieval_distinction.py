from __future__ import annotations

import pytest
from fuzzyxai.repository_diagnostics.auditor_v2 import (
    CalibrationObservation,
    EvidenceGroundedRouteAuditor,
    select_abstention_threshold,
)
from fuzzyxai.repository_diagnostics.graph import (
    EvidenceRef,
    RepositoryEdge,
    RepositoryGraph,
    RepositoryNode,
)


def test_global_ranking_promotes_shared_upstream_cause() -> None:
    nodes = [
        RepositoryNode(
            "upstream",
            "function",
            "fixture/repo",
            "src/upstream.py",
            "validate_schema",
            {"semantic_tokens": ("schema", "validate")},
        ),
    ]
    edges = []
    evidence = []
    obligations = []
    for index in range(2):
        obligation = f"failure-{index}"
        test = f"test-{index}"
        bridge = f"bridge-{index}"
        downstream = f"downstream-{index}"
        runtime = f"runtime-{index}"
        nodes.extend(
            (
                RepositoryNode(test, "test", "fixture/repo"),
                RepositoryNode(bridge, "module", "fixture/repo"),
                RepositoryNode(
                    downstream,
                    "function",
                    "fixture/repo",
                    f"src/downstream_{index}.py",
                    f"validate_{index}",
                    {"semantic_tokens": ("schema", "validate")},
                ),
                RepositoryNode(
                    runtime,
                    "runtime_exception",
                    "fixture/repo",
                    attributes={"obligation": obligation},
                ),
            )
        )
        trace_ref = f"trace-{index}"
        evidence.append(
            EvidenceRef(
                trace_ref,
                "traceback",
                f"src/downstream_{index}.py",
                f"line 1 in validate_{index}",
            )
        )
        edges.extend(
            (
                RepositoryEdge(f"fail-{index}", test, runtime, "fails_in"),
                RepositoryEdge(f"test-bridge-{index}", test, bridge, "calls"),
                RepositoryEdge(
                    f"bridge-upstream-{index}",
                    bridge,
                    "upstream",
                    "calls",
                ),
                RepositoryEdge(
                    f"trace-edge-{index}",
                    downstream,
                    runtime,
                    "produces",
                    (trace_ref,),
                ),
            )
        )
        obligations.append(obligation)
    graph = RepositoryGraph(
        "fixture/repo",
        "buggy",
        tuple(nodes),
        tuple(edges),
        tuple(evidence),
        tuple(obligations),
    )
    auditor = EvidenceGroundedRouteAuditor(abstention_threshold=0.0)
    greedy = auditor.audit(graph, "B_GREEDY")
    global_result = auditor.audit(graph, "O_ROUTE")
    assert greedy.candidates[0].node_id.startswith("downstream-")
    assert global_result.candidates[0].node_id == "upstream"
    assert global_result.status == "DIAGNOSIS_CONFIRMED"
    assert global_result.selected_cut == ("upstream",)
    assert tuple(item.node_id for item in greedy.candidates) != tuple(item.node_id for item in global_result.candidates)


def test_threshold_selection_preserves_registered_coverage() -> None:
    observations = tuple(CalibrationObservation(index / 10, index >= 3, index >= 3) for index in range(10))
    selected = select_abstention_threshold(
        observations,
        minimum_coverage=0.70,
    )
    assert selected.coverage >= 0.70
    assert selected.threshold == 0.3


def test_medium_confidence_returns_candidates_not_false_confirmation() -> None:
    graph = _shared_upstream_graph()
    result = EvidenceGroundedRouteAuditor(
        abstention_threshold=0.999,
        candidate_threshold=0.10,
    ).audit(graph, "O_ROUTE")
    assert result.status == "DIAGNOSIS_CANDIDATES"
    assert result.selected_cut == ()
    assert result.candidates


def test_low_confidence_returns_insufficient_evidence() -> None:
    graph = _shared_upstream_graph()
    result = EvidenceGroundedRouteAuditor(
        abstention_threshold=0.999,
        candidate_threshold=0.999,
    ).audit(graph, "O_ROUTE")
    assert result.status == "INSUFFICIENT_EVIDENCE"


def test_threshold_selection_does_not_count_ineligible_incidents() -> None:
    observations = tuple(CalibrationObservation(0.8, True, True, index < 6) for index in range(10))
    with pytest.raises(ValueError, match="minimum coverage"):
        select_abstention_threshold(
            observations,
            minimum_coverage=0.70,
        )


def _shared_upstream_graph() -> RepositoryGraph:
    nodes = [
        RepositoryNode(
            "upstream",
            "function",
            "fixture/repo",
            "src/upstream.py",
            "validate_schema",
            {"semantic_tokens": ("schema", "validate")},
        ),
        RepositoryNode("test", "test", "fixture/repo"),
        RepositoryNode(
            "runtime",
            "runtime_exception",
            "fixture/repo",
            attributes={"obligation": "failure"},
        ),
    ]
    return RepositoryGraph(
        "fixture/repo",
        "buggy",
        tuple(nodes),
        (
            RepositoryEdge("fail", "test", "runtime", "fails_in"),
            RepositoryEdge("call", "test", "upstream", "calls"),
        ),
        (),
        ("failure",),
    )
