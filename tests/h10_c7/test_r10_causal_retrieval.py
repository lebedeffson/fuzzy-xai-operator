from __future__ import annotations

import hashlib
import json
from pathlib import Path

from fuzzyxai.experiments.h10_c5c_runtime import _launcher_source
from fuzzyxai.experiments.h10_c7r_r10 import (
    audit_runtime_rows,
    development_gates,
    enrich_graph_with_source_excerpts,
    summarize_r10,
)
from fuzzyxai.repository_diagnostics.active_evidence import (
    R10TargetedProbePlanner,
)
from fuzzyxai.repository_diagnostics.graph import (
    RepositoryGraph,
    RepositoryNode,
)
from fuzzyxai.repository_diagnostics.guided_diagnosis import (
    GuidedNaturalDiagnosisEngine,
)
from fuzzyxai.repository_diagnostics.guided_retrieval import (
    R9_SOURCE_KINDS,
    FileRetriever,
    IncidentQuery,
    R10SourceAwareReranker,
    R10SymbolPoolBuilder,
    RankedSymbol,
    SymbolDocument,
    _runtime_profile,
    documents_from_graph,
)
from fuzzyxai.repository_diagnostics.runtime_events import (
    RuntimeEvent,
    normalize_runtime_event_rows,
)


def _document(
    node_id: str,
    file_path: str,
    symbol: str,
    text: str,
    **runtime: object,
) -> SymbolDocument:
    return SymbolDocument(
        node_id,
        file_path,
        symbol,
        text,
        **runtime,
    )


def _ranked(document: SymbolDocument, score: float) -> RankedSymbol:
    return RankedSymbol(
        document.node_id,
        document.file_path,
        document.symbol,
        score,
        ("fixture",),
        document.line_count,
        (),
    )


def test_runtime_rows_preserve_physical_order_and_occurrences() -> None:
    rows = [
        {
            "event_id": "hash-z",
            "test_id": "test",
            "kind": "call",
            "source_file": "src/a.py",
            "sequence_id": 8,
            "occurrence_count": 4,
        },
        {
            "event_id": "hash-a",
            "test_id": "test",
            "kind": "call",
            "source_file": "src/b.py",
            "sequence_id": 3,
        },
    ]
    normalized = normalize_runtime_event_rows(rows)
    assert [row["event_id"] for row in normalized] == ["hash-z", "hash-a"]
    assert [row["sequence_id"] for row in normalized] == [8, 9]
    assert normalized[0]["occurrence_count"] == 4


def test_runtime_profile_uses_sequence_and_occurrence_count() -> None:
    graph = RepositoryGraph(
        "repo",
        "buggy",
        (
            RepositoryNode("a", "function", "repo", "src/a.py", "a"),
            RepositoryNode("b", "function", "repo", "src/b.py", "b"),
        ),
        (),
        (),
        (),
    )
    events = (
        RuntimeEvent(
            "z",
            "test",
            "call",
            "src/a.py",
            "a",
            sequence_id=1,
            last_sequence_id=1,
            occurrence_count=7,
        ),
        RuntimeEvent(
            "a",
            "test",
            "call",
            "src/b.py",
            "b",
            sequence_id=20,
            last_sequence_id=20,
        ),
    )
    profile = _runtime_profile(graph, events)
    assert profile["a"]["execution_frequency"] == 7
    assert profile["b"]["last_touch_proximity"] == 1.0
    assert profile["a"]["last_touch_proximity"] < 0.1


def test_launcher_uses_ordered_tail_and_causal_event_types() -> None:
    source = _launcher_source(
        Path("/repo"),
        Path("/events"),
        "tests/test_x.py::test_x",
        ("pytest", "tests/test_x.py::test_x"),
    )
    compile(source, "runtime_launcher.py", "exec")
    assert "EVENTS = collections.deque()" in source
    assert "TAIL_LIMIT = 20000" in source
    assert "sorted(EVENTS.values()" not in source
    assert "'_aggregate_key': aggregate_key" in source
    assert "aggregate_key = (TEST_ID, kind, source_file" in source
    assert "aggregate_key = (TEST_ID, kind, source_file, source_symbol, target_file, target_symbol, detail)" not in source
    assert "sys.settrace(_trace)" in source
    assert "'.venv'" in source
    assert "'site-packages'" in source
    assert "object.__getattribute__(value, 'shape')" in source
    assert "shape = getattr(value, 'shape', None)" not in source
    for kind in (
        "argument_value",
        "return_value",
        "assertion_operand",
        "last_writer",
        "value_flow",
    ):
        assert repr(kind) in source


def test_file_first_retrieval_prefers_runtime_grounded_file() -> None:
    query = IncidentQuery(
        "incident",
        "wrong payload returned",
        ("tests/test_target.py::test_payload",),
        "AssertionError: payload",
        "expected payload",
    )
    documents = (
        _document("target", "src/target.py", "build_payload", "build payload"),
        _document("noise", "src/noise.py", "payload_log", "payload payload"),
    )
    events = (
        RuntimeEvent(
            "runtime",
            query.failing_tests[0],
            "last_writer",
            "src/target.py",
            "build_payload",
            sequence_id=10,
            last_sequence_id=10,
        ),
    )
    ranking = FileRetriever().rank(query, documents, events)
    assert ranking[0].file_path == "src/target.py"


def test_symbol_pool_respects_file_and_global_budgets() -> None:
    query = IncidentQuery("incident", "target", ("test_target",), "", "")
    documents = tuple(
        _document(
            f"{file_index}-{symbol_index}",
            f"src/f{file_index}.py",
            f"symbol_{symbol_index}",
            f"target symbol {symbol_index}",
        )
        for file_index in range(3)
        for symbol_index in range(12)
    )
    files = FileRetriever().rank(query, documents, limit=2)
    pool = R10SymbolPoolBuilder().build(
        query,
        files,
        documents,
        symbols_per_file=4,
        pool_limit=7,
    )
    assert len(pool) == 7
    assert len({item.file_path for item in pool}) <= 2
    assert all(item.file_path in {file.file_path for file in files} for item in pool)


def test_causal_last_writer_changes_symbol_order() -> None:
    first = _document("first", "src/a.py", "first", "first")
    writer = _document("writer", "src/b.py", "writer", "writer")
    ranking = (_ranked(first, 2.0), _ranked(writer, 1.0))
    event = RuntimeEvent(
        "writer-event",
        "test",
        "last_writer",
        "src/b.py",
        "writer",
        target_file="src/a.py",
        target_symbol="first",
        detail=json.dumps({"object_id": "run:x"}),
        sequence_id=4,
    )
    reranked = R10SourceAwareReranker().rerank(
        ranking,
        (first, writer),
        (event,),
    )
    assert reranked[0].node_id == "writer"
    assert "r10_causal_runtime" in reranked[0].rank_sources


def test_r10_semantic_variants_fail_closed_without_local_model(
    route_graph: RepositoryGraph,
) -> None:
    query = IncidentQuery(
        "incident",
        "shape mismatch",
        ("tests/test_loader.py::test_shape",),
        "ValueError",
    )
    result = GuidedNaturalDiagnosisEngine(structural_only=True).diagnose(
        route_graph,
        query,
        "R10B",
    )
    assert result.status == "VARIANT_UNAVAILABLE"
    assert result.unavailable_reason == "no_registered_source_aware_cross_encoder"


def test_r10_targeted_probe_is_bounded_to_two_requests() -> None:
    candidates = tuple(
        RankedSymbol(
            f"n{index}",
            f"src/n{index}.py",
            f"symbol_{index}",
            1.0 / (index + 1),
            ("fixture",),
            1,
            (),
        )
        for index in range(3)
    )
    requests = R10TargetedProbePlanner().plan("tests/test_x.py::test_x", candidates)
    assert len(requests) == 2
    assert all(request.safety_level == "READ_ONLY_TEST_EXECUTION" for request in requests)
    assert "FUZZYXAI_R10_TARGETS=" in requests[0].command[1]


def test_runtime_event_old_mapping_remains_compatible() -> None:
    event = RuntimeEvent.from_mapping(
        {
            "event_id": "old",
            "test_id": "test",
            "kind": "coverage",
            "source_file": "src/old.py",
        }
    )
    assert event.sequence_id == -1
    assert event.occurrence_count == 1


def test_r10_runtime_audit_rejects_reconstructed_old_events() -> None:
    readiness = audit_runtime_rows(
        (
            {
                "event_id": "old",
                "test_id": "test",
                "kind": "coverage",
                "source_file": "src/a.py",
            },
        )
    )
    assert readiness.status == "R10_RUNTIME_RECOLLECTION_REQUIRED"
    assert "sequence_id" in readiness.missing_fields
    assert not readiness.ready


def test_r10_runtime_audit_accepts_collected_causal_chronology() -> None:
    kinds = ("coverage", "call", "traceback_frame", "exception")
    rows = tuple(
        {
            "event_id": f"event-{sequence}",
            "sequence_id": sequence,
            "timestamp_ns": sequence + 1,
            "thread_id": 1,
            "call_depth": sequence,
            "test_id": "tests/test_x.py::test_x",
            "kind": kind,
            "source_file": "src/x.py",
            "occurrence_count": 1,
        }
        for sequence, kind in enumerate(kinds)
    )
    readiness = audit_runtime_rows(rows)
    assert readiness.ready
    assert readiness.has_core_runtime
    assert readiness.has_causal_observation
    assert readiness.max_sequence_id == 3
    assert readiness.full_tail_end_preserved


def test_r10_runtime_audit_rejects_hash_ordered_sequence() -> None:
    rows = (
        {
            "event_id": "z",
            "sequence_id": 2,
            "timestamp_ns": 2,
            "thread_id": 1,
            "call_depth": 0,
            "test_id": "test",
            "kind": "coverage",
            "source_file": "src/a.py",
            "occurrence_count": 1,
        },
        {
            "event_id": "a",
            "sequence_id": 1,
            "timestamp_ns": 3,
            "thread_id": 1,
            "call_depth": 0,
            "test_id": "test",
            "kind": "call",
            "source_file": "src/a.py",
            "occurrence_count": 1,
        },
    )
    readiness = audit_runtime_rows(rows)
    assert "row[1]:non_monotonic_sequence" in readiness.chronology_errors


def test_r10_protocol_matches_implementation_budgets() -> None:
    payload = json.loads(
        Path(
            "protocol/h10_c7r_r10/R10_DEVELOPMENT_PROTOCOL_LOCK.json"
        ).read_text(encoding="utf-8")
    )
    budgets = payload["candidate_budgets"]
    assert budgets == {
        "file_candidates": 20,
        "final_symbols": 20,
        "semantic_rerank_candidates": 100,
        "symbol_pool": 200,
        "symbols_per_file": 10,
    }
    assert payload["maximum_targeted_probes"] == 2
    assert payload["scientific_result"] == "NOT_EVALUATED"


def test_r10_release_provenance_separates_commit_and_ci_pairs() -> None:
    payload = json.loads(
        Path(
            "results/h10_c7r_r10/R10_RELEASE_PROVENANCE.json"
        ).read_text(encoding="utf-8")
    )
    assert payload["implementation_ci"]["head_sha"] == (
        payload["implementation_commit"]
    )
    assert payload["release_ci"]["head_sha"] == payload["release_commit"]
    assert not payload["code_changed_between_implementation_and_release"]
    assert all(
        path.startswith(("reports/h10_c7r_r10/", "results/h10_c7r_r10/"))
        for path in payload["changed_files"]
    )


def test_r10_parent_results_match_locked_sha256() -> None:
    payload = json.loads(
        Path("protocol/h10_c7r_r10/PARENT_IMMUTABILITY.json").read_text(
            encoding="utf-8"
        )
    )
    for raw_path, expected in payload["protected_sha256"].items():
        actual = hashlib.sha256(Path(raw_path).read_bytes()).hexdigest()
        assert actual == expected


def test_r10_development_gates_are_fail_closed() -> None:
    rows = [
        {
            "incident_id": f"i{index}",
            "repository": f"r{index}",
            "variant": "R10A",
            "available": True,
            "file_hit_at_10": 1.0,
            "file_hit_at_20": 1.0,
            "pool_hit_at_200": 1.0,
            "symbol_hit_at_20": float(index != 0),
            "contract_reordered_localization": False,
        }
        for index in range(4)
    ]
    summary = summarize_r10(rows)
    gates = development_gates(summary)
    assert summary["symbol_recall_at_20"] == 0.75
    assert not gates["symbol_recall_at_20_at_least_0_85"]
    assert not all(gates.values())


def test_r10_contract_inference_does_not_reorder_localization(
    route_graph: RepositoryGraph,
) -> None:
    query = IncidentQuery(
        "incident",
        "shape mismatch",
        ("tests/test_loader.py::test_shape",),
        "ValueError",
        "expected shape (2,) got (3,)",
    )
    engine = GuidedNaturalDiagnosisEngine(structural_only=True)
    documents = documents_from_graph(
        route_graph,
        (),
        source_kinds=R9_SOURCE_KINDS,
    )
    ranking, unavailable = engine._r10_ranking(
        query,
        documents,
        (),
        "R10A",
    )
    diagnosis = engine.diagnose(route_graph, query, "R10A")
    assert unavailable is None
    assert [item.node_id for item in diagnosis.candidates] == [
        item.node_id for item in ranking
    ]


def test_r10_source_excerpt_is_added_without_modifying_frozen_importer(
    tmp_path: Path,
) -> None:
    source = tmp_path / "src" / "sample.py"
    source.parent.mkdir()
    source.write_text(
        "def build_payload(value):\n    return {'value': value}\n",
        encoding="utf-8",
    )
    graph = RepositoryGraph(
        "repo",
        "buggy",
        (
            RepositoryNode(
                "node",
                "function",
                "repo",
                "src/sample.py",
                "build_payload",
                {"lineno": 1, "end_lineno": 2},
            ),
        ),
        (),
        (),
        (),
    )
    enriched = enrich_graph_with_source_excerpts(graph, tmp_path)
    assert "return {'value': value}" in str(
        enriched.nodes[0].attributes["source_excerpt"]
    )
