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
