from __future__ import annotations

from fuzzyxai.experiments.h10_c5c_posthoc import (
    _auditor,
    _interpret,
    _oracle_retrieved_pool,
    _OracleContractInferer,
)
from fuzzyxai.gold_repository import GoldRepairAtom, RepositoryGold
from fuzzyxai.repository_diagnostics.contract_inference import (
    EvidenceGroundedContractInferer,
)
from fuzzyxai.repository_diagnostics.graph import (
    RepositoryGraph,
    RepositoryNode,
)
from fuzzyxai.repository_diagnostics.retrieval import (
    CandidateFeatures,
    RetrievedCandidate,
)


def _fixture() -> tuple[
    RepositoryGraph,
    tuple[RetrievedCandidate, ...],
    RepositoryGold,
]:
    graph = RepositoryGraph(
        "fixture/repo",
        "buggy",
        (
            RepositoryNode(
                "file-gold",
                "file",
                "fixture/repo",
                "src/gold.py",
            ),
            RepositoryNode(
                "gold",
                "function",
                "fixture/repo",
                "src/gold.py",
                "validate_shape",
                {
                    "semantic_tokens": (
                        "dtype",
                        "shape",
                        "validate",
                    )
                },
            ),
            RepositoryNode(
                "wrong",
                "function",
                "fixture/repo",
                "src/wrong.py",
                "decode",
                {"semantic_tokens": ("decode", "loads")},
            ),
        ),
        (),
        (),
        ("failing_test:0:test_failure",),
    )
    observed = (
        RetrievedCandidate(
            "wrong",
            "fixture/repo",
            "src/wrong.py",
            "decode",
            10.0,
            0.9,
            graph.obligations,
            (),
            (),
            CandidateFeatures(),
        ),
    )
    gold = RepositoryGold(
        (
            GoldRepairAtom(
                "src/gold.py",
                "validate_shape",
                "DATA_CONTRACT",
                "align_data_schema",
            ),
        ),
        ("src/gold.py",),
        (("src/gold.py", "validate_shape"),),
    )
    return graph, observed, gold


def _hit(result: object, gold: RepositoryGold) -> bool:
    return any(
        candidate.file_path == atom.file_path
        and candidate.symbol == atom.symbol
        and candidate.contract == atom.contract
        for candidate in result.candidates
        for atom in gold.atoms
    )


def test_oracle_variants_isolate_candidate_and_contract_inputs() -> None:
    graph, observed, gold = _fixture()
    oracle_graph, oracle_pool = _oracle_retrieved_pool(
        graph,
        observed,
        gold,
    )

    baseline = _auditor(
        0.0,
        observed,
        EvidenceGroundedContractInferer(),
    ).audit(graph, "O_ROUTE")
    candidate = _auditor(
        0.0,
        oracle_pool,
        EvidenceGroundedContractInferer(),
    ).audit(oracle_graph, "O_ROUTE")
    contract = _auditor(
        0.0,
        observed,
        _OracleContractInferer(gold),
    ).audit(graph, "O_ROUTE")
    both = _auditor(
        0.0,
        oracle_pool,
        _OracleContractInferer(gold),
    ).audit(oracle_graph, "O_ROUTE")

    assert not _hit(baseline, gold)
    assert _hit(candidate, gold)
    assert not _hit(contract, gold)
    assert _hit(both, gold)
    assert {
        item.node_id for item in observed
    } == {"wrong"}
    assert {
        item.node_id for item in oracle_pool
    } == {"gold", "wrong"}
    injected = next(item for item in oracle_pool if item.node_id == "gold")
    assert injected.retrieval_score == 10.0
    assert injected.covered_obligations == graph.obligations


def test_oracle_candidate_replaces_low_rank_existing_gold() -> None:
    graph, observed, gold = _fixture()
    low_rank_gold = RetrievedCandidate(
        "gold",
        "fixture/repo",
        "src/gold.py",
        "validate_shape",
        0.1,
        0.01,
        (),
        (),
        (),
        CandidateFeatures(),
    )
    _, oracle_pool = _oracle_retrieved_pool(
        graph,
        (*observed, low_rank_gold),
        gold,
    )

    gold_candidates = [
        item
        for item in oracle_pool
        if item.file_path == "src/gold.py"
        and item.symbol == "validate_shape"
    ]
    assert len(gold_candidates) == 1
    assert gold_candidates[0].retrieval_score == 10.0
    assert gold_candidates[0].confidence == 0.9
    assert gold_candidates[0].covered_obligations == graph.obligations


def test_oracle_candidate_adds_gold_source_missing_from_graph() -> None:
    graph, observed, _ = _fixture()
    missing = RepositoryGold(
        (
            GoldRepairAtom(
                "src/missing.py",
                "missing_symbol",
                "DATA_CONTRACT",
                "align_data_schema",
            ),
        ),
        ("src/missing.py",),
        (("src/missing.py", "missing_symbol"),),
    )
    oracle_graph, oracle_pool = _oracle_retrieved_pool(
        graph,
        observed,
        missing,
    )

    candidate = next(
        item
        for item in oracle_pool
        if item.file_path == "src/missing.py"
    )
    assert candidate.symbol == "missing_symbol"
    assert oracle_graph.node(candidate.node_id) is not None


def test_oracle_contract_does_not_normalize_unregistered_gold() -> None:
    graph, observed, gold = _fixture()
    unregistered = RepositoryGold(
        (
            GoldRepairAtom(
                "src/gold.py",
                "validate_shape",
                "CONFIGURATION",
                "change_program_symbol",
            ),
        ),
        gold.changed_files,
        gold.changed_symbols,
    )
    inference = _OracleContractInferer(unregistered).infer(
        graph,
        observed[0],
    )
    assert inference.contract == "CONFIGURATION"
    assert inference.supported is False
    assert inference.confidence == 0.0


def test_posthoc_interpretation_is_descriptive() -> None:
    metrics = {
        "BASELINE": {"joint_hit_at_3": 0.0},
        "ORACLE_CANDIDATE": {"joint_hit_at_3": 0.2},
        "ORACLE_CONTRACT": {"joint_hit_at_3": 0.0},
        "ORACLE_BOTH": {"joint_hit_at_3": 0.2},
    }
    assert _interpret(metrics) == "RETRIEVAL_DOMINANT"
