from __future__ import annotations

from fuzzyxai.repository_diagnostics.graph import RepositoryGraph
from fuzzyxai.repository_diagnostics.guided_diagnosis import (
    GuidedNaturalDiagnosisEngine,
)
from fuzzyxai.repository_diagnostics.guided_retrieval import (
    CandidateReservoir,
    DenseRetriever,
    HashingCodeEncoder,
    IncidentNormalizer,
    IncidentQuery,
    RankedSymbol,
    RepoGraphRanker,
    SymbolDocument,
    documents_from_graph,
    reciprocal_rank_fusion,
    tokenize,
)


def _query() -> IncidentQuery:
    return IncidentQuery(
        "fixture",
        "schema shape mismatch while rendering decoded cache",
        ("tests/test_loader.py::test_schema_shape",),
        "ValueError in load_schema",
        "expected shape (4,), observed shape (3,)",
    )


def test_true_cause_is_retrieved_from_traceback(
    route_graph: RepositoryGraph,
) -> None:
    result = GuidedNaturalDiagnosisEngine().diagnose(
        route_graph,
        _query(),
        "R1",
    )
    assert result.candidates[0].node_id == "cause"
    assert result.status == "DIAGNOSIS_CANDIDATES"


def test_caller_of_traceback_symbol_remains_in_small_context(
    route_graph: RepositoryGraph,
) -> None:
    ranked = RepoGraphRanker().rank(
        route_graph,
        documents_from_graph(route_graph),
    )
    assert {item.node_id for item in ranked[:3]} >= {"cause", "symptom"}


def test_bm25_lexical_decoy_is_corrected_by_dynamic_graph(
    route_graph: RepositoryGraph,
) -> None:
    engine = GuidedNaturalDiagnosisEngine(
        dense_encoders=(
            HashingCodeEncoder(128),
            HashingCodeEncoder(256),
        )
    )
    result = engine.diagnose(route_graph, _query(), "R3")
    assert result.candidates[0].node_id == "cause"
    assert "lexical-decoy" in {
        item.node_id for item in result.candidates
    }


def test_dense_retrieval_finds_semantically_matching_symbol() -> None:
    documents = (
        SymbolDocument(
            "target",
            "src/io.py",
            "deserialize_payload",
            "deserialize payload reader schema",
        ),
        SymbolDocument(
            "decoy",
            "src/ui.py",
            "render",
            "render button window",
        ),
    )
    ranking = DenseRetriever(HashingCodeEncoder()).rank(
        "payload deserialize reader",
        documents,
    )
    assert ranking[0].node_id == "target"


def test_repo_graph_promotes_executed_upstream_neighbour(
    route_graph: RepositoryGraph,
) -> None:
    ranking = RepoGraphRanker().rank(
        route_graph,
        documents_from_graph(route_graph),
    )
    assert "cause" in {item.node_id for item in ranking[:2]}


def test_rrf_merges_independent_rank_sources() -> None:
    first = RankedSymbol("a", "a.py", "a", 4.0, ("bm25",), 1, ())
    second = RankedSymbol("a", "a.py", "a", 2.0, ("dense",), 1, ())
    other = RankedSymbol("b", "b.py", "b", 3.0, ("bm25",), 1, ())
    fused = reciprocal_rank_fusion(((first, other), (second,)))
    assert fused[0].node_id == "a"
    assert fused[0].rank_sources == ("bm25", "dense")


def test_incident_normalizer_removes_collection_noise() -> None:
    query = IncidentQuery(
        "fixture",
        (
            "Error processing line 1 of /tmp/run/site-packages/a.pth\n"
            "_distutils_hack failed\n"
            "real project failure"
        ),
        ("tests/test_loader.py::test_shape",),
        "/tmp/run/runtime_launcher.py failed\nValueError: wrong shape",
        "expected shape 4 observed shape 3",
    )
    normalized = IncidentNormalizer().normalize(query)
    assert "_distutils_hack" not in normalized.weighted_text
    assert "runtime_launcher.py" not in normalized.weighted_text
    assert "real project failure" in normalized.weighted_text
    assert "ValueError: wrong shape" in normalized.weighted_text


def test_identifier_tokenizer_splits_program_symbols() -> None:
    values = set(
        tokenize(
            "test_Host_header_overwrite "
            "FacebookIE StaticFileHandler.validate_absolute_path"
        )
    )
    assert {"host", "header", "overwrite", "facebook", "static", "path"} <= values


def test_candidate_reservoir_preserves_single_channel_candidate() -> None:
    decoys = tuple(
        RankedSymbol(
            f"decoy-{index}",
            f"src/decoy_{index}.py",
            f"decoy_{index}",
            1.0,
            ("bm25",),
            1,
            (),
        )
        for index in range(80)
    )
    cause = RankedSymbol(
        "cause",
        "src/cause.py",
        "cause",
        1.0,
        ("executed_slice",),
        1,
        (),
    )
    runtime_decoys = tuple(
        RankedSymbol(
            item.node_id,
            item.file_path,
            item.symbol,
            item.score,
            ("executed_slice",),
            item.line_count,
            (),
        )
        for item in decoys
    )
    values = CandidateReservoir().build(
        (
            decoys,
            (*runtime_decoys[:13], cause, *runtime_decoys[13:]),
        ),
        weights=(1.0, 1.35),
    )
    selected = next(item for item in values if item.node_id == "cause")
    assert "channel_rank:executed_slice:14" in selected.evidence


def test_global_order_prefers_one_shared_upstream_cause() -> None:
    upstream = RankedSymbol(
        "upstream",
        "src/upstream.py",
        "load",
        1.0,
        ("graph",),
        10,
        ("failure-a", "failure-b", "failure-c"),
    )
    downstream = tuple(
        RankedSymbol(
            f"downstream-{index}",
            f"src/downstream_{index}.py",
            "render",
            2.0,
            ("bm25",),
            10,
            (f"failure-{name}",),
        )
        for index, name in enumerate(("a", "b", "c"))
    )
    ranked = GuidedNaturalDiagnosisEngine._global_order(
        (*downstream, upstream),
        ("failure-a", "failure-b", "failure-c"),
    )
    assert ranked[0].node_id == "upstream"


def test_neural_variant_fails_closed_without_registered_encoder(
    route_graph: RepositoryGraph,
) -> None:
    result = GuidedNaturalDiagnosisEngine().diagnose(
        route_graph,
        _query(),
        "R2",
    )
    assert result.status == "VARIANT_UNAVAILABLE"
    assert result.unavailable_reason == "no_registered_dense_encoder"


def test_registered_baselines_are_reported_separately(
    route_graph: RepositoryGraph,
) -> None:
    engine = GuidedNaturalDiagnosisEngine(
        dense_encoders=(
            HashingCodeEncoder(128),
            HashingCodeEncoder(256),
        )
    )
    trace = engine.diagnose(route_graph, _query(), "B_TRACE")
    lexical = engine.diagnose(route_graph, _query(), "B_BM25")
    route = engine.diagnose(route_graph, _query(), "O_ROUTE")
    assert trace.variant == "B_TRACE"
    assert lexical.variant == "B_BM25"
    assert route.variant == "O_ROUTE"
    assert trace.candidates[0].node_id == "cause"
