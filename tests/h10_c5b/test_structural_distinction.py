from __future__ import annotations

from fuzzyxai.repository_diagnostics import audit_global, audit_greedy
from fuzzyxai.repository_diagnostics.graph import (
    EvidenceRef,
    RepositoryEdge,
    RepositoryGraph,
    RepositoryNode,
)

from .conftest import compositional_graph


def test_one_cause_one_symptom_methods_are_allowed_to_match() -> None:
    graph = RepositoryGraph(
        "fixture/repo",
        "buggy",
        (
            RepositoryNode(
                "source",
                "function",
                "fixture/repo",
                "src/source.py",
                "source",
                {"semantic_tokens": ("shape",)},
            ),
            RepositoryNode(
                "runtime",
                "runtime_exception",
                "fixture/repo",
                attributes={"obligation": "failing_test:0"},
            ),
        ),
        (RepositoryEdge("produces", "source", "runtime", "produces"),),
        (),
        ("failing_test:0",),
    )
    assert audit_global(graph).selected_cut == audit_greedy(graph).selected_cut


def test_upstream_cause_covering_three_failures_distinguishes_global() -> None:
    graph = compositional_graph(branches=3)
    greedy = audit_greedy(graph)
    global_result = audit_global(graph)
    assert greedy.selected_cut != global_result.selected_cut
    assert greedy.coverage == 1 / 3
    assert global_result.selected_cut == ("source",)
    assert global_result.coverage == 1.0


def test_two_local_symptoms_are_replaced_by_shared_upstream_cut() -> None:
    graph = compositional_graph(branches=2)
    assert audit_global(graph).selected_cut == ("source",)
    assert audit_global(graph).coverage == 1.0
    assert audit_greedy(graph).coverage == 0.5


def test_greedy_downstream_cut_fails_full_recertification_condition() -> None:
    graph = compositional_graph(branches=3)
    greedy = audit_greedy(graph)
    assert greedy.uncovered_obligations
    assert not (
        greedy.coverage == 1.0
        and not greedy.uncovered_obligations
    )
    assert not audit_global(graph).uncovered_obligations


def test_equivalent_global_optima_are_all_returned() -> None:
    result = audit_global(compositional_graph(branches=3, duplicate_source=True))
    assert result.selected_cut in {("source",), ("source-alt",)}
    assert {("source",), ("source-alt",)} <= set(result.equivalent_cuts)


def test_irrelevant_keyword_evidence_cannot_override_structure() -> None:
    graph = compositional_graph(branches=2)
    noisy = RepositoryGraph(
        graph.repository,
        graph.revision,
        (
            *graph.nodes,
            RepositoryNode(
                "noise",
                "configuration_key",
                graph.repository,
                "noise.toml",
                "pickle_version_schema",
                evidence_refs=("noise-evidence",),
            ),
        ),
        graph.edges,
        (
            *graph.evidence,
            EvidenceRef(
                "noise-evidence",
                "failing_test",
                "noise.toml",
                "pickle version shape dtype serialization",
            ),
        ),
        graph.obligations,
    )
    result = audit_global(noisy)
    assert result.selected_cut == ("source",)
    assert all(candidate.node_id != "noise" for candidate in result.candidates)


def test_exact_global_cover_can_select_more_than_three_independent_causes() -> None:
    nodes = []
    edges = []
    obligations = []
    for index in range(4):
        cause = f"cause-{index}"
        runtime = f"runtime-{index}"
        obligation = f"failure-{index}"
        nodes.extend(
            (
                RepositoryNode(
                    cause,
                    "function",
                    "fixture/repo",
                    f"src/cause_{index}.py",
                    f"cause_{index}",
                    {"semantic_tokens": ("shape",)},
                ),
                RepositoryNode(
                    runtime,
                    "runtime_exception",
                    "fixture/repo",
                    attributes={"obligation": obligation},
                ),
            )
        )
        edges.append(RepositoryEdge(f"produces-{index}", cause, runtime, "produces"))
        obligations.append(obligation)
    graph = RepositoryGraph(
        "fixture/repo",
        "buggy",
        tuple(nodes),
        tuple(edges),
        (),
        tuple(obligations),
    )
    result = audit_global(graph)
    assert result.coverage == 1.0
    assert result.selected_cut == ("cause-0", "cause-1", "cause-2", "cause-3")


def test_unknown_contract_evidence_causes_calibrated_abstention() -> None:
    graph = RepositoryGraph(
        "fixture/repo",
        "buggy",
        (
            RepositoryNode(
                "source",
                "function",
                "fixture/repo",
                "src/source.py",
                "opaque_operation",
            ),
            RepositoryNode(
                "runtime",
                "runtime_exception",
                "fixture/repo",
                attributes={"obligation": "failure"},
            ),
        ),
        (RepositoryEdge("produces", "source", "runtime", "produces"),),
        (),
        ("failure",),
    )
    for result in (audit_greedy(graph), audit_global(graph)):
        assert result.status == "INSUFFICIENT_EVIDENCE"
        assert "contract_family_not_supported_by_evidence" in result.limitations


def test_test_files_and_unregistered_imports_are_not_repair_candidates() -> None:
    graph = RepositoryGraph(
        "fixture/repo",
        "buggy",
        (
            RepositoryNode(
                "source",
                "function",
                "fixture/repo",
                "src/core.py",
                "deserialize",
                {"semantic_tokens": ("deserialize",)},
            ),
            RepositoryNode(
                "test-file",
                "file",
                "fixture/repo",
                "tests/test_core.py",
                evidence_refs=("failure",),
            ),
            RepositoryNode(
                "stdlib",
                "module",
                "fixture/repo",
                symbol="warnings",
                attributes={"external": True},
            ),
            RepositoryNode(
                "runtime",
                "runtime_exception",
                "fixture/repo",
                attributes={"obligation": "failure"},
            ),
        ),
        (
            RepositoryEdge("stdlib-import", "test-file", "stdlib", "imports"),
            RepositoryEdge("source-call", "test-file", "source", "calls"),
            RepositoryEdge("failure", "test-file", "runtime", "fails_in"),
        ),
        (EvidenceRef("failure", "failing_test", "test", "deserialize failed"),),
        ("failure",),
    )
    result = audit_global(graph)
    assert result.selected_cut == ("source",)
    assert all(candidate.node_id not in {"test-file", "stdlib"} for candidate in result.candidates)
