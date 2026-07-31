from __future__ import annotations

from pathlib import Path

import numpy as np
from fuzzyxai.experiments.h10_c7r_r10m import (
    BGE_RERANKER_ID,
    BGE_RERANKER_REVISION,
    GRAPHCODEBERT_ID,
    GRAPHCODEBERT_REVISION,
    R10MConfig,
    R10MRetriever,
    SQLiteModelCache,
    incident_model_text,
)
from fuzzyxai.repository_diagnostics.guided_retrieval import (
    IncidentQuery,
    SymbolDocument,
)
from fuzzyxai.repository_diagnostics.runtime_events import RuntimeEvent


class FixtureEncoder:
    model_name = GRAPHCODEBERT_ID
    revision = GRAPHCODEBERT_REVISION

    def encode(self, texts: list[str]) -> np.ndarray:
        return np.asarray(
            [
                [1.0, 0.0] if "target" in text.lower() else [0.0, 1.0]
                for text in texts
            ],
            dtype=np.float32,
        )


class FixtureReranker:
    model_name = BGE_RERANKER_ID
    revision = BGE_RERANKER_REVISION

    def score(self, query: str, texts: list[str]) -> tuple[float, ...]:
        assert "gold" not in query.lower()
        return tuple(float("target" in text.lower()) for text in texts)


def _documents() -> tuple[SymbolDocument, ...]:
    return (
        SymbolDocument(
            "target",
            "src/target.py",
            "build_target",
            "target payload builder",
            attributes={
                "node_kind": "function",
                "source_excerpt": "def build_target(): return 'target'",
            },
            executed=True,
            traceback_distance=0.0,
            dynamic_call_distance=0.0,
            last_touch_proximity=1.0,
        ),
        SymbolDocument(
            "same-file-decoy",
            "src/target.py",
            "log_payload",
            "logging helper",
            attributes={
                "node_kind": "function",
                "source_excerpt": "def log_payload(value): pass",
            },
        ),
        SymbolDocument(
            "other",
            "src/other.py",
            "other",
            "unrelated helper",
            attributes={
                "node_kind": "function",
                "source_excerpt": "def other(): return None",
            },
        ),
    )


def _events() -> tuple[RuntimeEvent, ...]:
    return (
        RuntimeEvent(
            "event-1",
            "tests/test_target.py::test_target",
            "last_writer",
            "src/target.py",
            "build_target",
            detail="target object",
            sequence_id=10,
            last_sequence_id=10,
        ),
    )


def test_r10m_uses_fixed_models_and_rrf_budget() -> None:
    assert GRAPHCODEBERT_REVISION == (
        "2b0488a7bb0eefc7041f1bb2cad1ab26b0da269d"
    )
    assert BGE_RERANKER_REVISION == (
        "953dc6f6f85a1b2dbfca4c34a2796e7dde08d41e"
    )
    assert R10MConfig().rrf_constant == 60
    assert R10MConfig().file_limit == 25
    assert R10MConfig().symbol_pool_limit == 200
    assert R10MConfig().final_limit == 20


def test_r10m_file_first_model_ranking_keeps_causal_symbol() -> None:
    query = IncidentQuery(
        "incident",
        "target payload is wrong",
        ("tests/test_target.py::test_target",),
        "AssertionError in build_target",
        "expected target",
    )
    result = R10MRetriever(FixtureEncoder(), FixtureReranker()).rank(
        query,
        _documents(),
        _events(),
    )
    assert result.top_files[0].file_path == "src/target.py"
    assert result.top_symbols[0].node_id == "target"
    assert len(result.symbol_pool) <= 200
    assert set(result.symbol_channel_ranks["target"]) == {
        "bge",
        "bm25",
        "causal",
        "graphcodebert",
    }


def test_incident_model_text_has_observable_channels_only() -> None:
    text = incident_model_text(
        IncidentQuery(
            "incident",
            "target failure",
            ("tests/test_target.py::test_target",),
            "AssertionError",
            "left != right",
        ),
        _events(),
    )
    assert "[ISSUE]" in text
    assert "[CAUSAL RUNTIME]" in text
    assert "gold_patch" not in text
    assert "fix_commit" not in text
    assert "changed_files" not in text


def test_sqlite_model_cache_round_trips_values(tmp_path: Path) -> None:
    cache = SQLiteModelCache(tmp_path / "cache.sqlite3")
    vector = np.asarray([0.25, 0.75], dtype=np.float32)
    cache.put_vector("vector", vector)
    cache.put_score("score", 1.5)
    assert np.allclose(cache.vector("vector"), vector)
    assert cache.score("score") == 1.5
