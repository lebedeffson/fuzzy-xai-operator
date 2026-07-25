from __future__ import annotations

from fuzzyxai.repository_diagnostics.graph import (
    EvidenceRef,
    RepositoryEdge,
    RepositoryGraph,
    RepositoryNode,
)


def compositional_graph(
    *,
    branches: int = 3,
    duplicate_source: bool = False,
) -> RepositoryGraph:
    nodes = [
        RepositoryNode(
            "source",
            "function",
            "fixture/repo",
            "src/source.py",
            "load_schema",
            {"semantic_tokens": ("shape", "schema")},
        ),
    ]
    if duplicate_source:
        nodes.append(
            RepositoryNode(
                "source-alt",
                "function",
                "fixture/repo",
                "src/alternative.py",
                "load_schema_alt",
                {"semantic_tokens": ("shape", "schema")},
            )
        )
    edges = []
    obligations = []
    evidence = []
    for index in range(branches):
        downstream = f"downstream-{index}"
        test = f"test-{index}"
        runtime = f"runtime-{index}"
        obligation = f"failing_test:{index}"
        nodes.extend(
            (
                RepositoryNode(
                    downstream,
                    "function",
                    "fixture/repo",
                    f"src/branch_{index}.py",
                    f"render_{index}",
                    {"semantic_tokens": ("shape",)},
                ),
                RepositoryNode(test, "test", "fixture/repo", f"tests/test_branch_{index}.py", f"test_branch_{index}"),
                RepositoryNode(
                    runtime,
                    "runtime_exception",
                    "fixture/repo",
                    symbol=f"test_branch_{index}",
                    attributes={"obligation": obligation},
                ),
            )
        )
        edges.extend(
            (
                RepositoryEdge(f"call-ds-source-{index}", downstream, "source", "calls"),
                RepositoryEdge(f"call-test-ds-{index}", test, downstream, "calls"),
                RepositoryEdge(f"failure-{index}", test, runtime, "fails_in"),
            )
        )
        if duplicate_source:
            edges.append(
                RepositoryEdge(f"call-ds-alt-{index}", downstream, "source-alt", "calls")
            )
        obligations.append(obligation)
        evidence.append(EvidenceRef(f"failure:{index}", "failing_test", test, obligation))
    return RepositoryGraph(
        "fixture/repo",
        "buggy",
        tuple(nodes),
        tuple(edges),
        tuple(evidence),
        tuple(obligations),
    )
