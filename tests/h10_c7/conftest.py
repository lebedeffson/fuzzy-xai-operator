from __future__ import annotations

import pytest
from fuzzyxai.repository_diagnostics.graph import (
    EvidenceRef,
    RepositoryEdge,
    RepositoryGraph,
    RepositoryNode,
)


@pytest.fixture
def route_graph() -> RepositoryGraph:
    return RepositoryGraph(
        "fixture/repository",
        "buggy",
        (
            RepositoryNode(
                "cause",
                "function",
                "fixture/repository",
                "src/loader.py",
                "load_schema",
                {
                    "line_count": 12,
                    "semantic_tokens": ("load", "schema", "shape"),
                    "source_excerpt": "validate incoming schema shape",
                },
            ),
            RepositoryNode(
                "symptom",
                "function",
                "fixture/repository",
                "src/render.py",
                "render_result",
                {
                    "line_count": 20,
                    "semantic_tokens": ("render", "result"),
                },
            ),
            RepositoryNode(
                "lexical-decoy",
                "function",
                "fixture/repository",
                "src/cache.py",
                "decode_cache",
                {
                    "line_count": 80,
                    "semantic_tokens": (
                        "decode",
                        "cache",
                        "shape",
                        "shape",
                        "shape",
                    ),
                },
            ),
            RepositoryNode("test", "test", "fixture/repository"),
            RepositoryNode(
                "runtime",
                "runtime_exception",
                "fixture/repository",
                symbol="tests/test_loader.py::test_schema_shape",
                attributes={"obligation": "shape_failure"},
            ),
        ),
        (
            RepositoryEdge("fail", "test", "runtime", "fails_in"),
            RepositoryEdge(
                "test-call",
                "test",
                "symptom",
                "runtime_calls",
                ("call-evidence",),
            ),
            RepositoryEdge("cause-call", "cause", "symptom", "calls"),
            RepositoryEdge(
                "trace",
                "cause",
                "runtime",
                "produces",
                ("trace-evidence",),
            ),
        ),
        (
            EvidenceRef(
                "trace-evidence",
                "traceback",
                "src/loader.py",
                "ValueError assertion shape mismatch in load_schema",
            ),
            EvidenceRef(
                "call-evidence",
                "runtime_call",
                "tests/test_loader.py",
                "test_schema_shape calls render_result",
            ),
        ),
        ("shape_failure",),
    )
