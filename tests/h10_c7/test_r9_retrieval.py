from __future__ import annotations

import hashlib
import json
from pathlib import Path

from fuzzyxai.experiments.h10_c7 import GoldAtom, GoldLocalization
from fuzzyxai.experiments.h10_c7a import BudgetCase
from fuzzyxai.experiments.h10_c7r_r9 import (
    LAMBDAMART_PARAMETERS,
    R9FeatureCase,
    fit_lambdamart,
    leave_one_repository_out,
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
    IncidentQuery,
    R9CandidateCompressor,
    R9CompressionConfig,
    RankedSymbol,
    StrictIdentifierExtractor,
    SymbolDocument,
    documents_from_graph,
)


def _ranked(
    node_id: str,
    channel: str,
    *,
    score: float = 1.0,
) -> RankedSymbol:
    return RankedSymbol(
        node_id,
        f"src/{node_id}.py",
        node_id,
        score,
        (channel,),
        1,
        (),
    )


def test_r9_source_schema_includes_previously_unreachable_kinds() -> None:
    assert {"module", "serialized_artifact", "runtime_symbol"} <= R9_SOURCE_KINDS
    graph = RepositoryGraph(
        "repo",
        "buggy",
        (
            RepositoryNode("module", "module", "repo", "pkg/module.py"),
            RepositoryNode(
                "artifact",
                "serialized_artifact",
                "repo",
                attributes={"artifact_path": "models/model.bin"},
            ),
            RepositoryNode(
                "runtime",
                "runtime_symbol",
                "repo",
                "pkg/runtime.py",
                "runtime_entry",
            ),
        ),
        (),
        (),
        (),
    )
    old = documents_from_graph(graph)
    r9 = documents_from_graph(graph, source_kinds=R9_SOURCE_KINDS)
    assert old == ()
    assert {item.node_id for item in r9} == {"module", "artifact", "runtime"}


def test_strict_identifier_does_not_treat_semantic_tokens_as_exact() -> None:
    query = IncidentQuery(
        "incident",
        "validate_absolute_path rejects traversal",
        ("tests/test_paths.py::test_traversal",),
        "ValueError in validate_absolute_path",
    )
    documents = (
        SymbolDocument(
            "target",
            "src/paths.py",
            "StaticFileHandler.validate_absolute_path",
            "validate absolute path",
        ),
        SymbolDocument(
            "decoy",
            "src/config.py",
            "get_value",
            "config data value",
            attributes={
                "semantic_tokens": (
                    "validate",
                    "absolute",
                    "path",
                    "traversal",
                )
            },
        ),
    )
    ranking = StrictIdentifierExtractor().rank(query, documents)
    assert [item.node_id for item in ranking] == ["target"]
    assert ranking[0].evidence[0].startswith("exact_identifier_level:")


def test_channel_quota_preserves_strong_bm25_candidate() -> None:
    documents = tuple(
        SymbolDocument(
            f"n{index}",
            f"src/n{index}.py",
            f"n{index}",
            f"symbol {index}",
            executed=index > 4,
            last_touch_proximity=1.0 if index > 4 else 0.0,
        )
        for index in range(30)
    )
    bm25 = tuple(_ranked(f"n{index}", "bm25") for index in range(5))
    runtime = tuple(
        _ranked(f"n{index}", "runtime", score=10.0)
        for index in range(5, 30)
    )
    ranking = R9CandidateCompressor().rank(
        {"bm25": bm25, "runtime": runtime},
        documents,
        hierarchical=False,
    )
    assert {item.node_id for item in ranking} >= {"n0", "n1", "n2", "n3", "n4"}


def test_runtime_v2_does_not_reward_hot_function() -> None:
    cold = SymbolDocument(
        "cold",
        "src/cold.py",
        "cold",
        "cold",
        executed=True,
        execution_frequency=1,
    )
    hot = SymbolDocument(
        "hot",
        "src/hot.py",
        "hot",
        "hot",
        executed=True,
        execution_frequency=1000,
    )
    ranking = GuidedNaturalDiagnosisEngine._runtime_ranking_v2((hot, cold))
    assert ranking[0].node_id == "cold"


def test_r9c_fails_closed_without_local_models(
    route_graph: RepositoryGraph,
) -> None:
    query = IncidentQuery(
        "fixture",
        "shape mismatch",
        ("tests/test_loader.py::test_shape",),
        "ValueError in load_schema",
    )
    result = GuidedNaturalDiagnosisEngine(structural_only=True).diagnose(
        route_graph,
        query,
        "R9C",
    )
    assert result.status == "VARIANT_UNAVAILABLE"
    assert result.unavailable_reason == "no_registered_dense_encoder"


def test_contract_inference_does_not_reorder_r9_top20(
    route_graph: RepositoryGraph,
) -> None:
    query = IncidentQuery(
        "fixture",
        "shape mismatch",
        ("tests/test_loader.py::test_shape",),
        "ValueError in load_schema",
    )
    engine = GuidedNaturalDiagnosisEngine(structural_only=True)
    documents = documents_from_graph(
        route_graph,
        source_kinds=R9_SOURCE_KINDS,
    )
    ranking, unavailable = engine._r9_ranking(
        route_graph,
        query,
        documents,
        "R9B",
    )
    result = engine.diagnose(route_graph, query, "R9B")
    assert unavailable is None
    assert [item.node_id for item in result.candidates] == [
        item.node_id for item in ranking
    ]
    assert all(
        "candidate_contract_pair" not in item.rank_sources
        for item in result.candidates
    )


def test_loro_selection_never_uses_held_repository() -> None:
    rows = [
        {
            "incident_id": f"{repository}-{variant}",
            "repository": repository,
            "variant": variant,
            "hit_at_20": hit,
            "schema_gap_count": 0,
            "contract_reordered": 0,
            "exact_candidate_count": 1,
        }
        for repository, values in {
            "a": {"R9A": 1.0, "R9B": 0.0},
            "b": {"R9A": 1.0, "R9B": 0.0},
            "c": {"R9A": 0.0, "R9B": 1.0},
        }.items()
        for variant, hit in values.items()
    ]
    selected, folds = leave_one_repository_out(rows)
    assert len(selected) == 3
    assert {fold["held_repository"] for fold in folds} == {"a", "b", "c"}
    fold_c = next(fold for fold in folds if fold["held_repository"] == "c")
    assert fold_c["selected_variant"] == "R9A"


def test_r9_protocol_limits_match_implementation() -> None:
    lock = json.loads(
        Path(
            "protocol/h10_c7r_r9/R9_DEVELOPMENT_PROTOCOL_LOCK.json"
        ).read_text(encoding="utf-8")
    )
    config = R9CompressionConfig()
    assert lock["channel_limits"] == {
        "bm25": config.bm25_limit,
        "dense": config.dense_limit,
        "graph": config.graph_limit,
        "legacy": config.legacy_limit,
        "runtime": config.runtime_limit,
        "strict_identifier": config.strict_identifier_limit,
    }


def test_h10_c7r_v1_files_remain_hash_locked() -> None:
    lock = json.loads(
        Path(
            "protocol/h10_c7r_r9/H10_C7R_V1_IMMUTABILITY.json"
        ).read_text(encoding="utf-8")
    )
    for name, expected in lock["protected_sha256"].items():
        actual = hashlib.sha256(Path(name).read_bytes()).hexdigest()
        assert actual == expected


def test_lambdamart_ranker_is_deterministic() -> None:
    compressor = R9CandidateCompressor()
    payloads = tuple(
        _feature_case(f"repo-{index}", legacy_support=index % 2 == 0)
        for index in range(8)
    )
    first = fit_lambdamart(payloads, compressor=compressor)
    second = fit_lambdamart(payloads, compressor=compressor)
    assert first.model_sha256 == second.model_sha256
    assert first.score(payloads[0].feature_rows) == second.score(
        payloads[0].feature_rows
    )
    assert not {
        "repository",
        "incident_id",
        "gold_symbol",
    }.intersection(first.feature_names)


def test_lambdamart_protocol_parameters_are_locked() -> None:
    lock = json.loads(
        Path(
            "protocol/h10_c7r_r9/R9_DEVELOPMENT_PROTOCOL_LOCK.json"
        ).read_text(encoding="utf-8")
    )
    ranker = lock["r9b_ranker"]
    assert ranker["backend_version"] == "4.6.0"
    assert ranker["n_estimators"] == LAMBDAMART_PARAMETERS["n_estimators"]
    assert ranker["random_state"] == LAMBDAMART_PARAMETERS["random_state"]
    assert ranker["training_split"] == "leave_one_repository_out"


def _feature_case(repository: str, *, legacy_support: bool) -> R9FeatureCase:
    graph = RepositoryGraph(repository, "buggy", (), (), (), ())
    query = IncidentQuery(
        f"{repository}-incident",
        "failure in target",
        ("tests/test_target.py::test_target",),
        "AssertionError",
    )
    case = BudgetCase(
        query.incident_id,
        repository,
        query,
        graph,
        (),
        2,
        2,
    )
    documents = (
        SymbolDocument("gold", "src/target.py", "target", "target"),
        SymbolDocument("decoy", "src/decoy.py", "decoy", "decoy"),
    )
    channels = {
        "bm25": (_ranked("decoy", "bm25"), _ranked("gold", "bm25")),
        "legacy": (
            (_ranked("gold", "legacy"),)
            if legacy_support
            else (_ranked("decoy", "legacy"),)
        ),
    }
    compressor_features = R9CandidateCompressor().feature_rows(
        channels, documents
    )
    return R9FeatureCase(
        case,
        GoldLocalization(
            case.incident_id,
            (GoldAtom("src/target.py", "target", "CONFIGURATION"),),
        ),
        documents,
        channels,
        compressor_features,
        0,
        0,
        0,
    )
