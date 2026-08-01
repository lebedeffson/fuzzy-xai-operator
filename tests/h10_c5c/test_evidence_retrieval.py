from __future__ import annotations

from fuzzyxai.repository_diagnostics.graph import (
    EvidenceRef,
    RepositoryEdge,
    RepositoryGraph,
    RepositoryNode,
)
from fuzzyxai.repository_diagnostics.retrieval import (
    EvidenceGroundedCandidateRetriever,
)


def test_exact_traceback_frame_beats_unrelated_keyword_node() -> None:
    graph = RepositoryGraph(
        "fixture/repo",
        "buggy",
        (
            RepositoryNode(
                "target",
                "function",
                "fixture/repo",
                "src/core.py",
                "validate_shape",
                {"semantic_tokens": ("shape", "validate")},
            ),
            RepositoryNode(
                "noise",
                "function",
                "fixture/repo",
                "src/noise.py",
                "deserialize_shape_version",
                {
                    "semantic_tokens": (
                        "deserialize",
                        "dtype",
                        "shape",
                        "version",
                    )
                },
            ),
            RepositoryNode(
                "runtime",
                "runtime_exception",
                "fixture/repo",
                attributes={"obligation": "failure"},
            ),
        ),
        (
            RepositoryEdge(
                "trace",
                "target",
                "runtime",
                "produces",
                ("trace-ref",),
            ),
        ),
        (
            EvidenceRef(
                "trace-ref",
                "traceback",
                "src/core.py",
                "line 10 in validate_shape",
            ),
        ),
        ("failure",),
    )
    candidates = EvidenceGroundedCandidateRetriever().retrieve(graph)
    assert [candidate.node_id for candidate in candidates] == ["target"]
    assert candidates[0].signals[0].kind == "exact_traceback_frame"


def test_dynamic_coverage_retrieves_executed_symbol_only() -> None:
    graph = RepositoryGraph(
        "fixture/repo",
        "buggy",
        (
            RepositoryNode(
                "executed",
                "function",
                "fixture/repo",
                "src/core.py",
                "transform",
                {"semantic_tokens": ("shape",)},
            ),
            RepositoryNode(
                "not-executed",
                "function",
                "fixture/repo",
                "src/core.py",
                "other",
                {"semantic_tokens": ("shape",)},
            ),
        ),
        (),
        (
            EvidenceRef(
                "coverage",
                "dynamic_coverage",
                "src/core.py",
                "executed symbol transform lines 8-12",
            ),
        ),
        ("failure",),
    )
    candidates = EvidenceGroundedCandidateRetriever().retrieve(graph)
    assert [candidate.node_id for candidate in candidates] == ["executed"]
