from __future__ import annotations

from fuzzyxai.repository_diagnostics.active_evidence import (
    ActiveEvidenceRequestPlanner,
    apply_probe_observation,
)
from fuzzyxai.repository_diagnostics.contract_inference_v2 import (
    HierarchicalContractInferenceEngine,
)
from fuzzyxai.repository_diagnostics.graph import RepositoryGraph, RepositoryNode
from fuzzyxai.repository_diagnostics.guided_diagnosis import (
    GuidedNaturalDiagnosisEngine,
)
from fuzzyxai.repository_diagnostics.guided_retrieval import (
    IncidentQuery,
    RankedSymbol,
)
from fuzzyxai.repository_diagnostics.repair_validation import (
    RepairExecutionEvidence,
    classify_repair_execution,
)
from fuzzyxai.repository_diagnostics.runtime_events import RuntimeEvent


def _candidate(node_id: str = "cause") -> RankedSymbol:
    return RankedSymbol(
        node_id,
        "src/loader.py",
        "load_schema",
        0.5,
        ("fixture",),
        12,
        ("shape_failure",),
    )


def _query(text: str) -> IncidentQuery:
    return IncidentQuery(
        "fixture",
        text,
        ("tests/test_loader.py::test_schema_shape",),
        text,
        text,
    )


def test_assertion_diff_infers_data_shape_contract(
    route_graph: RepositoryGraph,
) -> None:
    result = HierarchicalContractInferenceEngine().infer(
        _query("expected shape 4 but observed shape 3"),
        _candidate(),
        route_graph,
    )
    assert result[0].family == "DATA_SHAPE"


def test_unknown_contract_is_explicit(
) -> None:
    graph = RepositoryGraph(
        "fixture/repository",
        "buggy",
        (
            RepositoryNode(
                "opaque",
                "function",
                "fixture/repository",
                "src/core.py",
                "opaque",
            ),
        ),
        (),
        (),
        ("opaque_failure",),
    )
    result = HierarchicalContractInferenceEngine().infer(
        _query("opaque failure"),
        RankedSymbol(
            "opaque",
            "src/core.py",
            "opaque",
            0.5,
            ("fixture",),
            1,
            ("opaque_failure",),
        ),
        graph,
    )
    assert result[0].family == "UNKNOWN_CONTRACT"
    assert result[0].confidence == 0.0


def test_behavioral_contract_maps_to_published_configuration_family(
    route_graph: RepositoryGraph,
) -> None:
    result = HierarchicalContractInferenceEngine().infer(
        _query("500 != 400 response status code"),
        RankedSymbol(
            "cause",
            "src/websocket.py",
            "accept_connection",
            0.5,
            ("traceback",),
            12,
            ("protocol_failure",),
        ),
        route_graph,
    )
    assert result[0].family == "PROTOCOL_RESPONSE"


def test_evidence_request_distinguishes_close_hypotheses() -> None:
    requests = ActiveEvidenceRequestPlanner().plan(
        "tests/test_loader.py::test_schema_shape",
        (_candidate("a"), _candidate("b")),
    )
    assert requests
    assert requests[0].safety_level == "READ_ONLY_TEST_EXECUTION"
    assert set(requests[0].distinguished_hypotheses) == {"a", "b"}


def test_registered_probe_observation_improves_true_rank() -> None:
    candidates = (
        _candidate("decoy"),
        RankedSymbol(
            "cause",
            "src/loader.py",
            "load_schema",
            0.4,
            ("fixture",),
            12,
            ("shape_failure",),
        ),
    )
    reranked = apply_probe_observation(
        candidates,
        ("cause",),
        confidence=0.8,
    )
    assert reranked[0].node_id == "cause"
    assert "registered_probe_observation" in reranked[0].rank_sources


def test_replay_probe_uses_saved_traceback_observation(
    route_graph: RepositoryGraph,
) -> None:
    candidates = (
        RankedSymbol(
            "decoy",
            "src/decoy.py",
            "render",
            1.0,
            ("bm25",),
            1,
            (),
        ),
        _candidate("cause"),
    )
    event = RuntimeEvent(
        "probe",
        "tests/test_loader.py::test_schema_shape",
        "traceback_frame",
        "src/loader.py",
        "load_schema",
    )
    reranked, status, details = (
        GuidedNaturalDiagnosisEngine._apply_replay_probe(
            route_graph,
            candidates,
            (event,),
        )
    )
    assert status == "ACTIVE_EVIDENCE_APPLIED"
    assert reranked[0].node_id == "cause"
    assert dict(details)["rank_before"] == "2"


def test_repair_requires_all_verification_stages() -> None:
    evidence = RepairExecutionEvidence(
        True,
        True,
        True,
        True,
        True,
        0,
        True,
    )
    assert classify_repair_execution(evidence) == "RESTORATION_CONFIRMED"


def test_regression_failure_blocks_recertification_claim() -> None:
    evidence = RepairExecutionEvidence(
        True,
        True,
        True,
        False,
        True,
        0,
        True,
    )
    assert classify_repair_execution(evidence) == "REGRESSION_FAILED"


def test_new_critical_violation_blocks_restoration() -> None:
    evidence = RepairExecutionEvidence(
        True,
        True,
        True,
        True,
        True,
        1,
        True,
    )
    assert classify_repair_execution(evidence) == "RECERTIFICATION_FAILED"
