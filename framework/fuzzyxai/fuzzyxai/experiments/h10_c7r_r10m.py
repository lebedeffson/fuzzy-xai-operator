from __future__ import annotations

import hashlib
import os
import sqlite3
from collections import defaultdict
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

import numpy as np

from fuzzyxai.repository_diagnostics.guided_retrieval import (
    BM25Retriever,
    IncidentNormalizer,
    IncidentQuery,
    RankedFile,
    RankedSymbol,
    SymbolDocument,
    reciprocal_rank_fusion,
)
from fuzzyxai.repository_diagnostics.runtime_events import RuntimeEvent

GRAPHCODEBERT_ID = "microsoft/graphcodebert-base"
GRAPHCODEBERT_REVISION = "2b0488a7bb0eefc7041f1bb2cad1ab26b0da269d"
BGE_RERANKER_ID = "BAAI/bge-reranker-v2-m3"
BGE_RERANKER_REVISION = "953dc6f6f85a1b2dbfca4c34a2796e7dde08d41e"


class TextEncoder(Protocol):
    model_name: str
    revision: str

    def encode(self, texts: Sequence[str]) -> np.ndarray:
        ...


class PairScorer(Protocol):
    model_name: str
    revision: str

    def score(self, query: str, texts: Sequence[str]) -> Sequence[float]:
        ...


@dataclass(frozen=True)
class R10MConfig:
    rrf_constant: int = 60
    file_limit: int = 25
    symbol_pool_limit: int = 200
    final_limit: int = 20
    graphcodebert_max_length: int = 256
    graphcodebert_batch_size: int = 24
    bge_max_length: int = 512
    bge_batch_size: int = 8
    runtime_pool_limit: int = 40
    traceback_pool_limit: int = 30
    value_flow_pool_limit: int = 30
    dense_pool_limit: int = 50
    bm25_pool_limit: int = 40
    graph_pool_limit: int = 10


@dataclass(frozen=True)
class R10MRanking:
    top_files: tuple[RankedFile, ...]
    symbol_pool: tuple[RankedSymbol, ...]
    top_symbols: tuple[RankedSymbol, ...]
    file_channel_ranks: dict[str, dict[str, int]]
    symbol_channel_ranks: dict[str, dict[str, int]]


class SQLiteModelCache:
    """Small persistent cache keyed by frozen model, parameters, and text."""

    def __init__(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        self.connection = sqlite3.connect(path)
        self.connection.execute(
            "CREATE TABLE IF NOT EXISTS vectors "
            "(cache_key TEXT PRIMARY KEY, dimensions INTEGER, payload BLOB)"
        )
        self.connection.execute(
            "CREATE TABLE IF NOT EXISTS scores "
            "(cache_key TEXT PRIMARY KEY, value REAL)"
        )
        self.connection.commit()

    def vector(self, key: str) -> np.ndarray | None:
        row = self.connection.execute(
            "SELECT dimensions, payload FROM vectors WHERE cache_key = ?",
            (key,),
        ).fetchone()
        if row is None:
            return None
        value = np.frombuffer(row[1], dtype=np.float32)
        if value.size != int(row[0]):
            raise RuntimeError("corrupt R10M vector cache entry")
        return value.copy()

    def put_vector(self, key: str, value: np.ndarray) -> None:
        vector = np.asarray(value, dtype=np.float32)
        self.connection.execute(
            "INSERT OR REPLACE INTO vectors VALUES (?, ?, ?)",
            (key, int(vector.size), vector.tobytes()),
        )
        self.connection.commit()

    def score(self, key: str) -> float | None:
        row = self.connection.execute(
            "SELECT value FROM scores WHERE cache_key = ?",
            (key,),
        ).fetchone()
        return float(row[0]) if row is not None else None

    def put_score(self, key: str, value: float) -> None:
        self.connection.execute(
            "INSERT OR REPLACE INTO scores VALUES (?, ?)",
            (key, float(value)),
        )
        self.connection.commit()


class FrozenGraphCodeBERT:
    """Pinned GraphCodeBERT inference with local-only loading and disk cache."""

    model_name = GRAPHCODEBERT_ID
    revision = GRAPHCODEBERT_REVISION

    def __init__(
        self,
        local_path: Path,
        cache: SQLiteModelCache,
        *,
        max_length: int = 256,
        batch_size: int = 24,
        device: str = "cuda",
        precision: str = "float16",
    ) -> None:
        self.local_path = local_path.resolve()
        self.cache = cache
        self.max_length = max_length
        self.batch_size = batch_size
        self.device_name = device
        self.precision = precision
        self._tokenizer = None
        self._model = None

    def _load(self) -> None:
        if self._model is not None:
            return
        _force_offline()
        _disable_transformers_vision_backend()
        import torch
        from transformers import AutoModel, AutoTokenizer

        if self.device_name == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("R10M model lock requires CUDA but CUDA is unavailable")
        self._tokenizer = AutoTokenizer.from_pretrained(
            self.local_path,
            local_files_only=True,
        )
        self._model = AutoModel.from_pretrained(
            self.local_path,
            local_files_only=True,
        )
        self._model.eval().to(self.device_name)
        if self.precision == "float16" and self.device_name == "cuda":
            self._model.half()

    def encode(self, texts: Sequence[str]) -> np.ndarray:
        if not texts:
            return np.empty((0, 0), dtype=np.float32)
        keys = [self._key(text) for text in texts]
        values = [self.cache.vector(key) for key in keys]
        missing = [index for index, value in enumerate(values) if value is None]
        if missing:
            self._load()
            import torch

            assert self._tokenizer is not None
            assert self._model is not None
            for start in range(0, len(missing), self.batch_size):
                indices = missing[start : start + self.batch_size]
                encoded = self._tokenizer(
                    [texts[index] for index in indices],
                    padding=True,
                    truncation=True,
                    max_length=self.max_length,
                    return_tensors="pt",
                )
                encoded = {
                    name: value.to(self.device_name)
                    for name, value in encoded.items()
                }
                with torch.inference_mode():
                    hidden = self._model(**encoded).last_hidden_state
                    mask = encoded["attention_mask"].unsqueeze(-1)
                    pooled = (hidden * mask).sum(1) / mask.sum(1).clamp_min(1)
                    pooled = torch.nn.functional.normalize(pooled.float(), dim=1)
                for index, vector in zip(indices, pooled.cpu().numpy()):
                    values[index] = vector
                    self.cache.put_vector(keys[index], vector)
        return np.stack([value for value in values if value is not None])

    def _key(self, text: str) -> str:
        payload = (
            f"{self.model_name}@{self.revision}|{self.max_length}|{text}"
        ).encode()
        return hashlib.sha256(payload).hexdigest()


class FrozenBGEReranker:
    """Pinned BGE pair scoring with local-only loading and disk cache."""

    model_name = BGE_RERANKER_ID
    revision = BGE_RERANKER_REVISION

    def __init__(
        self,
        local_path: Path,
        cache: SQLiteModelCache,
        *,
        max_length: int = 512,
        batch_size: int = 8,
        device: str = "cuda",
        precision: str = "float16",
    ) -> None:
        self.local_path = local_path.resolve()
        self.cache = cache
        self.max_length = max_length
        self.batch_size = batch_size
        self.device_name = device
        self.precision = precision
        self._tokenizer = None
        self._model = None

    def _load(self) -> None:
        if self._model is not None:
            return
        _force_offline()
        _disable_transformers_vision_backend()
        import torch
        from transformers import AutoModelForSequenceClassification, AutoTokenizer

        if self.device_name == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("R10M model lock requires CUDA but CUDA is unavailable")
        self._tokenizer = AutoTokenizer.from_pretrained(
            self.local_path,
            local_files_only=True,
        )
        self._model = AutoModelForSequenceClassification.from_pretrained(
            self.local_path,
            local_files_only=True,
        )
        self._model.eval().to(self.device_name)
        if self.precision == "float16" and self.device_name == "cuda":
            self._model.half()

    def score(self, query: str, texts: Sequence[str]) -> tuple[float, ...]:
        keys = [self._key(query, text) for text in texts]
        values = [self.cache.score(key) for key in keys]
        missing = [index for index, value in enumerate(values) if value is None]
        if missing:
            self._load()
            import torch

            assert self._tokenizer is not None
            assert self._model is not None
            for start in range(0, len(missing), self.batch_size):
                indices = missing[start : start + self.batch_size]
                encoded = self._tokenizer(
                    [query for _ in indices],
                    [texts[index] for index in indices],
                    padding=True,
                    truncation=True,
                    max_length=self.max_length,
                    return_tensors="pt",
                )
                encoded = {
                    name: value.to(self.device_name)
                    for name, value in encoded.items()
                }
                with torch.inference_mode():
                    logits = self._model(**encoded).logits.float().reshape(-1)
                for index, value in zip(indices, logits.cpu().tolist()):
                    values[index] = float(value)
                    self.cache.put_score(keys[index], float(value))
        return tuple(float(value) for value in values if value is not None)

    def _key(self, query: str, text: str) -> str:
        payload = (
            f"{self.model_name}@{self.revision}|{self.max_length}|"
            f"{query}\0{text}"
        ).encode()
        return hashlib.sha256(payload).hexdigest()


class R10MRetriever:
    """Frozen causal, lexical, dense, and pairwise model retrieval."""

    def __init__(
        self,
        encoder: TextEncoder,
        reranker: PairScorer,
        config: R10MConfig | None = None,
    ) -> None:
        self.encoder = encoder
        self.reranker = reranker
        self.config = config or R10MConfig()
        self.bm25 = BM25Retriever()

    def rank(
        self,
        query: IncidentQuery,
        documents: Sequence[SymbolDocument],
        runtime_events: Sequence[RuntimeEvent],
    ) -> R10MRanking:
        incident_text = incident_model_text(query, runtime_events)
        file_documents = _file_documents(documents)
        file_channels = {
            "causal": _runtime_file_ranking(file_documents, runtime_events),
            "trace": _trace_file_ranking(file_documents),
            "bm25": self.bm25.rank(
                incident_text,
                file_documents,
                limit=len(file_documents),
            ),
            "graphcodebert": _dense_ranking(
                incident_text,
                file_documents,
                self.encoder,
            ),
        }
        fused_files = reciprocal_rank_fusion(
            tuple(file_channels.values()),
            k=self.config.rrf_constant,
            limit=self.config.file_limit,
        )
        top_files = tuple(
            RankedFile(
                item.file_path,
                item.score,
                item.rank_sources,
                sum(
                    document.file_path == item.file_path
                    for document in documents
                ),
                sum(
                    document.line_count
                    for document in documents
                    if document.file_path == item.file_path
                ),
            )
            for item in fused_files
        )
        selected_paths = {item.file_path for item in top_files}
        local_documents = tuple(
            item for item in documents if item.file_path in selected_paths
        )
        pool_channels = {
            "runtime": _runtime_symbol_ranking(local_documents, runtime_events),
            "trace": _trace_symbol_ranking(local_documents),
            "value_flow": _value_flow_symbol_ranking(
                local_documents,
                runtime_events,
            ),
            "graphcodebert": _dense_ranking(
                incident_text,
                local_documents,
                self.encoder,
            ),
            "bm25": self.bm25.rank(
                incident_text,
                local_documents,
                limit=len(local_documents),
            ),
            "graph": _graph_distance_ranking(local_documents),
        }
        pool = _priority_union(
            (
                pool_channels["runtime"][: self.config.runtime_pool_limit],
                pool_channels["trace"][: self.config.traceback_pool_limit],
                pool_channels["value_flow"][
                    : self.config.value_flow_pool_limit
                ],
                pool_channels["graphcodebert"][
                    : self.config.dense_pool_limit
                ],
                pool_channels["bm25"][: self.config.bm25_pool_limit],
                pool_channels["graph"][: self.config.graph_pool_limit],
            ),
            limit=self.config.symbol_pool_limit,
        )
        by_id = {document.node_id: document for document in local_documents}
        pool_documents = tuple(by_id[item.node_id] for item in pool)
        final_channels = {
            "causal": _runtime_symbol_ranking(pool_documents, runtime_events),
            "bm25": self.bm25.rank(
                incident_text,
                pool_documents,
                limit=len(pool_documents),
            ),
            "graphcodebert": _dense_ranking(
                incident_text,
                pool_documents,
                self.encoder,
            ),
            "bge": _pair_ranking(
                incident_text,
                pool_documents,
                self.reranker,
            ),
        }
        top_symbols = reciprocal_rank_fusion(
            tuple(final_channels.values()),
            k=self.config.rrf_constant,
            limit=self.config.final_limit,
        )
        return R10MRanking(
            top_files=top_files,
            symbol_pool=pool,
            top_symbols=top_symbols,
            file_channel_ranks=_channel_ranks(file_channels),
            symbol_channel_ranks=_channel_ranks(final_channels),
        )


def incident_model_text(
    query: IncidentQuery,
    runtime_events: Sequence[RuntimeEvent],
) -> str:
    normalized = IncidentNormalizer().normalize(query)
    causal = [
        event
        for event in runtime_events
        if event.kind
        in {
            "argument_value",
            "assertion_operand",
            "exception",
            "last_writer",
            "return_value",
            "value_flow",
        }
    ][-64:]
    runtime = "\n".join(
        " | ".join(
            (
                event.kind,
                event.source_file or "",
                event.source_symbol or "",
                event.target_file or "",
                event.target_symbol or "",
                event.detail[:500],
            )
        )
        for event in causal
    )
    return "\n".join(
        (
            "[ASSERTION]",
            normalized.assertion[:1600],
            "[EXCEPTION]",
            normalized.exception[:1000],
            "[FAILING TEST]",
            normalized.failing_tests[:1000],
            "[PROJECT TRACEBACK]",
            normalized.traceback[-3000:],
            "[CAUSAL RUNTIME]",
            runtime[-3000:],
            "[ISSUE]",
            normalized.issue[:3000],
        )
    )


def candidate_model_text(document: SymbolDocument) -> str:
    attributes = document.attributes
    return "\n".join(
        (
            f"[FILE] {document.file_path}",
            f"[SYMBOL] {document.symbol or '<module>'}",
            f"[KIND] {attributes.get('node_kind', '')}",
            f"[SIGNATURE] {attributes.get('parameters', '')}",
            f"[RETURN] {attributes.get('return_annotation', '')}",
            "[SOURCE]",
            str(attributes.get("source_excerpt", document.text))[:12000],
            "[RUNTIME]",
            (
                f"executed={document.executed}; "
                f"traceback_distance={document.traceback_distance}; "
                f"call_distance={document.dynamic_call_distance}; "
                f"last_touch={document.last_touch_proximity:.8f}"
            ),
        )
    )


def _file_documents(
    documents: Sequence[SymbolDocument],
) -> tuple[SymbolDocument, ...]:
    by_file: dict[str, list[SymbolDocument]] = defaultdict(list)
    for document in documents:
        by_file[document.file_path].append(document)
    values = []
    for file_path, members in sorted(by_file.items()):
        ordered = sorted(members, key=lambda item: item.symbol or "")
        symbol_names = " ".join(item.symbol or "" for item in ordered)
        excerpts = "\n".join(
            str(item.attributes.get("source_excerpt", ""))[:1200]
            for item in ordered[:12]
        )
        values.append(
            SymbolDocument(
                node_id=f"r10m-file:{file_path}",
                file_path=file_path,
                symbol=None,
                text=f"{file_path}\n{symbol_names}\n{excerpts}",
                line_count=sum(item.line_count for item in members),
                executed=any(item.executed for item in members),
                traceback_distance=_minimum(
                    item.traceback_distance for item in members
                ),
                dynamic_call_distance=_minimum(
                    item.dynamic_call_distance for item in members
                ),
                execution_frequency=sum(
                    item.execution_frequency for item in members
                ),
                last_touch_proximity=max(
                    (item.last_touch_proximity for item in members),
                    default=0.0,
                ),
                failing_test_frequency=max(
                    (item.failing_test_frequency for item in members),
                    default=0.0,
                ),
            )
        )
    return tuple(values)


def _dense_ranking(
    query: str,
    documents: Sequence[SymbolDocument],
    encoder: TextEncoder,
) -> tuple[RankedSymbol, ...]:
    if not documents:
        return ()
    texts = [candidate_model_text(item) for item in documents]
    vectors = encoder.encode([query, *texts])
    if len(vectors) != len(documents) + 1:
        raise RuntimeError("GraphCodeBERT returned an invalid embedding count")
    query_vector = vectors[0]
    scores = np.asarray(vectors[1:]) @ np.asarray(query_vector)
    values = [
        _ranked(document, float(score), "graphcodebert")
        for document, score in zip(documents, scores)
    ]
    return tuple(
        sorted(
            values,
            key=lambda item: (-item.score, item.file_path, item.symbol or ""),
        )
    )


def _pair_ranking(
    query: str,
    documents: Sequence[SymbolDocument],
    scorer: PairScorer,
) -> tuple[RankedSymbol, ...]:
    scores = scorer.score(
        query,
        [candidate_model_text(item) for item in documents],
    )
    if len(scores) != len(documents):
        raise RuntimeError("BGE returned an invalid score count")
    return tuple(
        sorted(
            (
                _ranked(document, float(score), "bge_reranker")
                for document, score in zip(documents, scores)
            ),
            key=lambda item: (-item.score, item.file_path, item.symbol or ""),
        )
    )


def _runtime_file_ranking(
    documents: Sequence[SymbolDocument],
    events: Sequence[RuntimeEvent],
) -> tuple[RankedSymbol, ...]:
    latest = max(
        (event.last_sequence_id for event in events),
        default=0,
    )
    scores: dict[str, float] = defaultdict(float)
    for event in events:
        sequence = max(event.sequence_id, event.last_sequence_id)
        proximity = 1.0 / (1.0 + max(0, latest - sequence))
        kind = 2.0 if event.kind in {"last_writer", "value_flow"} else 1.0
        for file_path in (event.source_file, event.target_file):
            if file_path:
                scores[file_path] = max(scores[file_path], kind + proximity)
    return _rank_documents_by_file_score(documents, scores, "causal_file")


def _trace_file_ranking(
    documents: Sequence[SymbolDocument],
) -> tuple[RankedSymbol, ...]:
    values = [
        _ranked(
            item,
            3.0 * float(item.traceback_distance == 0.0)
            + 2.0 * float(item.executed)
            + _distance(item.dynamic_call_distance),
            "trace_execution_file",
        )
        for item in documents
        if item.executed or item.traceback_distance == 0.0
    ]
    return tuple(sorted(values, key=_rank_key))


def _runtime_symbol_ranking(
    documents: Sequence[SymbolDocument],
    events: Sequence[RuntimeEvent],
) -> tuple[RankedSymbol, ...]:
    causal = _event_symbol_scores(events, causal_only=False)
    values = []
    for item in documents:
        score = causal.get((item.file_path, item.symbol), 0.0)
        score += 2.0 * item.last_touch_proximity
        score += 1.5 * float(item.traceback_distance == 0.0)
        score += _distance(item.dynamic_call_distance)
        if score:
            values.append(_ranked(item, score, "causal"))
    return tuple(sorted(values, key=_rank_key))


def _value_flow_symbol_ranking(
    documents: Sequence[SymbolDocument],
    events: Sequence[RuntimeEvent],
) -> tuple[RankedSymbol, ...]:
    causal = _event_symbol_scores(events, causal_only=True)
    values = [
        _ranked(item, causal[(item.file_path, item.symbol)], "value_flow")
        for item in documents
        if (item.file_path, item.symbol) in causal
    ]
    return tuple(sorted(values, key=_rank_key))


def _trace_symbol_ranking(
    documents: Sequence[SymbolDocument],
) -> tuple[RankedSymbol, ...]:
    values = [
        _ranked(
            item,
            3.0 * float(item.traceback_distance == 0.0)
            + 2.0 * float(item.executed)
            + _distance(item.dynamic_call_distance),
            "trace_execution",
        )
        for item in documents
        if item.executed or item.traceback_distance == 0.0
    ]
    return tuple(sorted(values, key=_rank_key))


def _graph_distance_ranking(
    documents: Sequence[SymbolDocument],
) -> tuple[RankedSymbol, ...]:
    values = [
        _ranked(
            item,
            _distance(item.dynamic_call_distance)
            + _distance(item.directed_caller_distance)
            + _distance(item.directed_callee_distance),
            "graph_distance",
        )
        for item in documents
        if any(
            value is not None
            for value in (
                item.dynamic_call_distance,
                item.directed_caller_distance,
                item.directed_callee_distance,
            )
        )
    ]
    return tuple(sorted(values, key=_rank_key))


def _event_symbol_scores(
    events: Sequence[RuntimeEvent],
    *,
    causal_only: bool,
) -> dict[tuple[str, str | None], float]:
    accepted = (
        {"last_writer", "value_flow", "assertion_operand"}
        if causal_only
        else {
            "argument_value",
            "assertion_operand",
            "call",
            "exception",
            "last_writer",
            "return_value",
            "traceback_frame",
            "value_flow",
        }
    )
    latest = max((event.last_sequence_id for event in events), default=0)
    scores: dict[tuple[str, str | None], float] = defaultdict(float)
    for event in events:
        if event.kind not in accepted:
            continue
        proximity = 1.0 / (
            1.0
            + max(
                0,
                latest - max(event.sequence_id, event.last_sequence_id),
            )
        )
        kind_bonus = 2.0 if event.kind in {"last_writer", "value_flow"} else 1.0
        for file_path, symbol, direction in (
            (event.source_file, event.source_symbol, 1.0),
            (event.target_file, event.target_symbol, 0.25),
        ):
            if file_path:
                scores[(file_path, symbol)] += direction * (kind_bonus + proximity)
    return dict(scores)


def _priority_union(
    rankings: Sequence[Sequence[RankedSymbol]],
    *,
    limit: int,
) -> tuple[RankedSymbol, ...]:
    selected: list[RankedSymbol] = []
    seen: set[str] = set()
    for ranking in rankings:
        for item in ranking:
            if item.node_id in seen:
                continue
            selected.append(item)
            seen.add(item.node_id)
            if len(selected) == limit:
                return tuple(selected)
    return tuple(selected)


def _channel_ranks(
    channels: dict[str, Sequence[RankedSymbol]],
) -> dict[str, dict[str, int]]:
    values: dict[str, dict[str, int]] = defaultdict(dict)
    for channel, ranking in channels.items():
        for rank, item in enumerate(ranking, start=1):
            values[item.node_id][channel] = rank
    return dict(values)


def _rank_documents_by_file_score(
    documents: Sequence[SymbolDocument],
    scores: dict[str, float],
    source: str,
) -> tuple[RankedSymbol, ...]:
    return tuple(
        sorted(
            (
                _ranked(item, scores[item.file_path], source)
                for item in documents
                if item.file_path in scores
            ),
            key=_rank_key,
        )
    )


def _ranked(
    document: SymbolDocument,
    score: float,
    source: str,
) -> RankedSymbol:
    return RankedSymbol(
        document.node_id,
        document.file_path,
        document.symbol,
        score,
        (source,),
        document.line_count,
        document.obligations,
    )


def _rank_key(item: RankedSymbol) -> tuple[float, str, str]:
    return (-item.score, item.file_path, item.symbol or "")


def _distance(value: float | None) -> float:
    return 1.0 / (1.0 + value) if value is not None else 0.0


def _minimum(values: Iterable[float | None]) -> float | None:
    observed = [value for value in values if value is not None]
    return min(observed) if observed else None


def _force_offline() -> None:
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"


def _disable_transformers_vision_backend() -> None:
    """Keep text-only inference independent of an optional torchvision build."""
    import transformers.utils
    import transformers.utils.import_utils

    transformers.utils.is_torchvision_available = lambda: False
    transformers.utils.import_utils.is_torchvision_available = lambda: False
