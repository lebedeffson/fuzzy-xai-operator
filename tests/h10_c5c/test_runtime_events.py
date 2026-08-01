from __future__ import annotations

import json
from pathlib import Path

import pytest
from fuzzyxai.repository_diagnostics.graph import (
    RepositoryEdge,
    RepositoryGraph,
    RepositoryNode,
)
from fuzzyxai.repository_diagnostics.retrieval import (
    EvidenceGroundedCandidateRetriever,
)
from fuzzyxai.repository_diagnostics.runtime_events import (
    RuntimeEvent,
    RuntimeEvidenceAugmenter,
    load_runtime_events,
)


def _graph() -> RepositoryGraph:
    return RepositoryGraph(
        "fixture/repo",
        "buggy",
        (
            RepositoryNode(
                "test-a",
                "test",
                "fixture/repo",
                "tests/test_core.py",
                "tests/test_core.py::test_a",
            ),
            RepositoryNode(
                "test-b",
                "test",
                "fixture/repo",
                "tests/test_core.py",
                "tests/test_core.py::test_b",
            ),
            RepositoryNode(
                "source",
                "function",
                "fixture/repo",
                "src/core.py",
                "validate",
                {"semantic_tokens": ("shape", "validate")},
            ),
            RepositoryNode(
                "runtime-a",
                "runtime_exception",
                "fixture/repo",
                symbol="tests/test_core.py::test_a",
                attributes={"obligation": "failure-a"},
            ),
            RepositoryNode(
                "runtime-b",
                "runtime_exception",
                "fixture/repo",
                symbol="tests/test_core.py::test_b",
                attributes={"obligation": "failure-b"},
            ),
        ),
        (
            RepositoryEdge("fail-a", "test-a", "runtime-a", "fails_in"),
            RepositoryEdge("fail-b", "test-b", "runtime-b", "fails_in"),
            RepositoryEdge("broad-a", "source", "runtime-a", "produces"),
            RepositoryEdge("broad-b", "source", "runtime-b", "produces"),
        ),
        (),
        ("failure-a", "failure-b"),
    )


def test_per_test_runtime_coverage_is_not_broadcast_to_other_failures() -> None:
    event = RuntimeEvent(
        "event-a",
        "tests/test_core.py::test_a",
        "coverage",
        "src/core.py",
        "validate",
        detail="validate lines 1-5",
    )
    graph = RuntimeEvidenceAugmenter().apply(_graph(), (event,))
    candidate = EvidenceGroundedCandidateRetriever().retrieve(graph)[0]
    assert candidate.node_id == "source"
    assert candidate.covered_obligations == ("failure-a",)
    assert "broad-a" not in {edge.edge_id for edge in graph.edges}
    assert "broad-b" in {edge.edge_id for edge in graph.edges}


@pytest.mark.parametrize(
    ("test_id", "graph_symbol"),
    (
        ("tests/test_core.py::test_a", "test_a"),
        ("tests/test_core.py::TestCore::test_a[param]", "TestCore.test_a"),
    ),
)
def test_runtime_coverage_resolves_pytest_node_id_to_ast_test(
    test_id: str,
    graph_symbol: str,
) -> None:
    graph = _graph()
    nodes = tuple(
        RepositoryNode(
            node.node_id,
            node.kind,
            node.repository,
            node.file_path,
            graph_symbol if node.node_id == "test-a" else node.symbol,
            node.attributes,
            node.evidence_refs,
        )
        for node in graph.nodes
    )
    runtime_nodes = tuple(
        RepositoryNode(
            node.node_id,
            node.kind,
            node.repository,
            node.file_path,
            test_id if node.node_id == "runtime-a" else node.symbol,
            node.attributes,
            node.evidence_refs,
        )
        for node in nodes
    )
    event = RuntimeEvent(
        "event-a",
        test_id,
        "coverage",
        "src/core.py",
        "validate",
    )

    augmented = RuntimeEvidenceAugmenter().apply(
        RepositoryGraph(
            graph.repository,
            graph.revision,
            runtime_nodes,
            graph.edges,
            graph.evidence,
            graph.obligations,
            graph.limitations,
        ),
        (event,),
    )

    assert any(
        edge.source == "test-a"
        and edge.target == "source"
        and edge.relation == "executes"
        for edge in augmented.edges
    )


def test_runtime_coverage_registers_observed_non_ast_symbol() -> None:
    graph = _graph()
    graph = RepositoryGraph(
        graph.repository,
        graph.revision,
        (
            *graph.nodes,
            RepositoryNode(
                "source-file",
                "file",
                "fixture/repo",
                "src/core.py",
            ),
        ),
        graph.edges,
        graph.evidence,
        graph.obligations,
        graph.limitations,
    )
    event = RuntimeEvent(
        "event-a",
        "tests/test_core.py::test_a",
        "coverage",
        "src/core.py",
        "<setcomp>",
    )

    augmented = RuntimeEvidenceAugmenter().apply(graph, (event,))

    runtime_symbol = next(
        node
        for node in augmented.nodes
        if node.kind == "runtime_symbol"
    )
    assert runtime_symbol.file_path == "src/core.py"
    assert runtime_symbol.symbol == "<setcomp>"
    assert runtime_symbol.attributes["observed_at_runtime"] is True
    assert any(
        edge.source == "test-a"
        and edge.target == runtime_symbol.node_id
        and edge.relation == "executes"
        for edge in augmented.edges
    )


def test_runtime_symbol_requires_registered_source_file() -> None:
    event = RuntimeEvent(
        "event-a",
        "tests/test_core.py::test_a",
        "coverage",
        "src/missing.py",
        "<setcomp>",
    )
    with pytest.raises(ValueError, match="source is absent from graph"):
        RuntimeEvidenceAugmenter().apply(_graph(), (event,))


def test_runtime_only_test_support_is_recorded_but_not_retrieved() -> None:
    event = RuntimeEvent(
        "event-a",
        "tests/test_core.py::test_a",
        "coverage",
        "tests/__init__.py",
    )

    augmented = RuntimeEvidenceAugmenter().apply(_graph(), (event,))

    support = next(
        node
        for node in augmented.nodes
        if node.kind == "runtime_test_support"
    )
    assert support.file_path == "tests/__init__.py"
    assert support.attributes["source_unavailable"] is True
    assert EvidenceGroundedCandidateRetriever().retrieve(augmented) == ()


def test_runtime_event_stream_rejects_gold_fields(tmp_path: Path) -> None:
    path = tmp_path / "runtime.jsonl"
    path.write_text(
        json.dumps(
            {
                "event_id": "event",
                "test_id": "test",
                "kind": "coverage",
                "source_file": "src/core.py",
                "changed_files": ["src/core.py"],
            }
        )
        + "\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="Gold fields"):
        load_runtime_events(path)


def test_runtime_event_rejects_unknown_failing_test() -> None:
    event = RuntimeEvent(
        "event",
        "tests/test_core.py::unknown",
        "coverage",
        "src/core.py",
        "validate",
    )
    with pytest.raises(ValueError, match="unregistered failing tests"):
        RuntimeEvidenceAugmenter().apply(_graph(), (event,))
