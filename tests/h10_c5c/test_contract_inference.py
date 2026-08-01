from __future__ import annotations

from fuzzyxai.repository_diagnostics.contract_inference import (
    EvidenceGroundedContractInferer,
)
from fuzzyxai.repository_diagnostics.graph import (
    EvidenceRef,
    RepositoryGraph,
    RepositoryNode,
)
from fuzzyxai.repository_diagnostics.retrieval import (
    RetrievalSignal,
    RetrievedCandidate,
)


def _candidate(node_id: str) -> RetrievedCandidate:
    return RetrievedCandidate(
        node_id,
        "fixture/repo",
        "src/core.py",
        "validate_shape",
        14.0,
        0.8,
        ("failure",),
        ("trace",),
        (RetrievalSignal("exact_traceback_frame", 14.0, "failure", ("trace",)),),
    )


def test_type_assertion_and_schema_tokens_infer_data_contract() -> None:
    graph = RepositoryGraph(
        "fixture/repo",
        "buggy",
        (
            RepositoryNode(
                "source",
                "function",
                "fixture/repo",
                "src/core.py",
                "validate_shape",
                {"semantic_tokens": ("dtype", "shape", "validate")},
            ),
        ),
        (),
        (
            EvidenceRef(
                "trace",
                "traceback",
                "src/core.py",
                "ValueError assertion shape mismatch",
            ),
        ),
        ("failure",),
    )
    result = EvidenceGroundedContractInferer().infer(graph, _candidate("source"))
    assert result.contract == "DATA_CONTRACT"
    assert result.supported is True
    assert "schema_type_shape_tokens" in result.evidence_reasons


def test_opaque_symbol_fails_closed() -> None:
    graph = RepositoryGraph(
        "fixture/repo",
        "buggy",
        (RepositoryNode("source", "function", "fixture/repo", "src/core.py", "x"),),
        (),
        (),
        ("failure",),
    )
    result = EvidenceGroundedContractInferer().infer(graph, _candidate("source"))
    assert result.contract == "UNREGISTERED_CONTRACT"
    assert result.supported is False


def test_contract_engine_returns_ranked_alternatives() -> None:
    graph = RepositoryGraph(
        "fixture/repo",
        "buggy",
        (
            RepositoryNode(
                "source",
                "function",
                "fixture/repo",
                "src/core.py",
                "load_shape",
                {"semantic_tokens": ("load", "shape", "validate")},
            ),
        ),
        (),
        (
            EvidenceRef(
                "trace",
                "traceback",
                "src/core.py",
                "ValueError shape mismatch while deserialize",
            ),
        ),
        ("failure",),
    )
    ranked = EvidenceGroundedContractInferer().infer_candidates(
        graph,
        _candidate("source"),
    )
    assert ranked[0].family == "DATA_CONTRACT"
    assert {item.family for item in ranked} >= {
        "DATA_CONTRACT",
        "SERIALIZATION",
    }
