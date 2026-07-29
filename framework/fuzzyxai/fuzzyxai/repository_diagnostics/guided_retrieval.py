from __future__ import annotations

import hashlib
import math
import re
from collections import Counter, defaultdict
from collections.abc import Sequence
from dataclasses import dataclass, field
from itertools import pairwise
from typing import ClassVar, Protocol

from .graph import RepositoryGraph

TOKEN_RE = re.compile(r"[A-Za-z_][A-Za-z0-9_]{1,}|[0-9]+")
SOURCE_KINDS = frozenset({"class", "configuration_key", "dependency", "file", "function", "method"})
TRACEBACK_KINDS = frozenset({"runtime_exception", "runtime_traceback_frame", "traceback"})


def tokenize(value: str) -> tuple[str, ...]:
    return tuple(token.lower() for token in TOKEN_RE.findall(value))


@dataclass(frozen=True)
class IncidentQuery:
    incident_id: str
    issue: str
    failing_tests: tuple[str, ...]
    traceback: str
    assertion: str = ""

    @property
    def text(self) -> str:
        return " ".join(
            (self.issue, *self.failing_tests, self.traceback, self.assertion)
        )


@dataclass(frozen=True)
class SymbolDocument:
    node_id: str
    file_path: str
    symbol: str | None
    text: str
    line_count: int = 1
    executed: bool = False
    traceback_distance: float | None = None
    dynamic_call_distance: float | None = None
    obligations: tuple[str, ...] = ()
    attributes: dict[str, object] = field(default_factory=dict)


@dataclass(frozen=True)
class RankedSymbol:
    node_id: str
    file_path: str
    symbol: str | None
    score: float
    rank_sources: tuple[str, ...]
    line_count: int
    obligations: tuple[str, ...]
    evidence: tuple[str, ...] = ()


class DenseCodeEncoder(Protocol):
    model_name: str
    revision: str

    def encode(self, texts: Sequence[str]) -> Sequence[Sequence[float]]:
        ...


class HashingCodeEncoder:
    """Deterministic smoke backend; never reported as a neural model."""

    model_name = "fuzzyxai-hashing-code-encoder"
    revision = "1"

    def __init__(self, dimensions: int = 256) -> None:
        self.dimensions = dimensions

    def encode(self, texts: Sequence[str]) -> tuple[tuple[float, ...], ...]:
        return tuple(self._encode(text) for text in texts)

    def _encode(self, text: str) -> tuple[float, ...]:
        values = [0.0] * self.dimensions
        tokens = tokenize(text)
        features = (
            *tokens,
            *(f"{first}::{second}" for first, second in pairwise(tokens)),
        )
        for token in features:
            digest = hashlib.sha256(token.encode()).digest()
            index = int.from_bytes(digest[:4], "big") % self.dimensions
            sign = 1.0 if digest[4] & 1 else -1.0
            values[index] += sign
        norm = math.sqrt(sum(value * value for value in values))
        if norm:
            values = [value / norm for value in values]
        return tuple(values)


class LocalTransformerCodeEncoder:
    """Pinned local-only transformer backend with no network fallback."""

    def __init__(self, model_name: str, revision: str) -> None:
        self.model_name = model_name
        self.revision = revision

    def encode(self, texts: Sequence[str]) -> Sequence[Sequence[float]]:
        try:
            import torch
            from transformers import AutoModel, AutoTokenizer
        except ImportError as exc:
            raise RuntimeError("transformers backend is not installed") from exc
        tokenizer = AutoTokenizer.from_pretrained(
            self.model_name,
            revision=self.revision,
            local_files_only=True,
        )
        model = AutoModel.from_pretrained(
            self.model_name,
            revision=self.revision,
            local_files_only=True,
        )
        model.eval()
        output: list[list[float]] = []
        with torch.no_grad():
            for text in texts:
                encoded = tokenizer(
                    text,
                    truncation=True,
                    max_length=256,
                    return_tensors="pt",
                )
                hidden = model(**encoded).last_hidden_state
                mask = encoded["attention_mask"].unsqueeze(-1)
                pooled = (hidden * mask).sum(1) / mask.sum(1).clamp_min(1)
                vector = pooled[0]
                vector = vector / vector.norm().clamp_min(1e-12)
                output.append(vector.cpu().tolist())
        return output


def documents_from_graph(graph: RepositoryGraph) -> tuple[SymbolDocument, ...]:
    evidence = " ".join(item.detail for item in graph.evidence)
    call_edges = {
        edge.target
        for edge in graph.edges
        if edge.relation in {"runtime_calls", "tested_by"}
    }
    traceback_nodes = {
        edge.source
        for edge in graph.edges
        if edge.relation == "produces"
        and any(
            item.evidence_id in edge.evidence_refs
            and item.kind in TRACEBACK_KINDS
            for item in graph.evidence
        )
    }
    obligations_by_node: dict[str, set[str]] = defaultdict(set)
    for edge in graph.edges:
        target = graph.node(edge.target)
        if target and target.kind == "runtime_exception":
            obligation = target.attributes.get("obligation")
            if obligation:
                obligations_by_node[edge.source].add(str(obligation))
    documents = []
    for node in graph.nodes:
        if node.kind not in SOURCE_KINDS or not node.file_path:
            continue
        semantic = " ".join(
            str(value)
            for value in node.attributes.get("semantic_tokens", ())
        )
        text = " ".join(
            (
                node.file_path,
                node.symbol or "",
                node.kind,
                semantic,
                str(node.attributes.get("source_excerpt", "")),
                evidence if node.node_id in traceback_nodes else "",
            )
        )
        documents.append(
            SymbolDocument(
                node.node_id,
                node.file_path,
                node.symbol,
                text,
                max(1, int(node.attributes.get("line_count", 1))),
                node.node_id in call_edges or node.node_id in traceback_nodes,
                0.0 if node.node_id in traceback_nodes else None,
                0.0 if node.node_id in call_edges else None,
                tuple(sorted(obligations_by_node[node.node_id])),
                dict(node.attributes),
            )
        )
    return tuple(documents)


class BM25Retriever:
    def rank(
        self,
        query: str,
        documents: Sequence[SymbolDocument],
        *,
        limit: int = 50,
    ) -> tuple[RankedSymbol, ...]:
        if not documents:
            return ()
        query_tokens = tokenize(query)
        tokenized = [tokenize(document.text) for document in documents]
        average_length = sum(map(len, tokenized)) / max(1, len(tokenized))
        frequencies = Counter(
            token for tokens in tokenized for token in set(tokens)
        )
        scored = []
        for document, tokens in zip(documents, tokenized):
            counts = Counter(tokens)
            score = 0.0
            for token in query_tokens:
                count = counts[token]
                if not count:
                    continue
                inverse = math.log(
                    1.0
                    + (len(documents) - frequencies[token] + 0.5)
                    / (frequencies[token] + 0.5)
                )
                denominator = count + 1.5 * (
                    1.0 - 0.75 + 0.75 * len(tokens) / max(average_length, 1.0)
                )
                score += inverse * (count * 2.5 / denominator)
            if score:
                scored.append(
                    RankedSymbol(
                        document.node_id,
                        document.file_path,
                        document.symbol,
                        score,
                        ("bm25",),
                        document.line_count,
                        document.obligations,
                    )
                )
        return tuple(
            sorted(
                scored,
                key=lambda item: (-item.score, item.file_path, item.symbol or ""),
            )[:limit]
        )


class DenseRetriever:
    def __init__(self, encoder: DenseCodeEncoder) -> None:
        self.encoder = encoder

    def rank(
        self,
        query: str,
        documents: Sequence[SymbolDocument],
        *,
        limit: int = 50,
    ) -> tuple[RankedSymbol, ...]:
        if not documents:
            return ()
        vectors = self.encoder.encode([query, *(item.text for item in documents)])
        query_vector = vectors[0]
        scored = []
        for document, vector in zip(documents, vectors[1:]):
            score = sum(a * b for a, b in zip(query_vector, vector))
            if score <= 0:
                continue
            scored.append(
                RankedSymbol(
                    document.node_id,
                    document.file_path,
                    document.symbol,
                    score,
                    (f"dense:{self.encoder.model_name}@{self.encoder.revision}",),
                    document.line_count,
                    document.obligations,
                )
            )
        return tuple(
            sorted(
                scored,
                key=lambda item: (-item.score, item.file_path, item.symbol or ""),
            )[:limit]
        )


class RepoGraphRanker:
    def rank(
        self,
        graph: RepositoryGraph,
        documents: Sequence[SymbolDocument],
        *,
        limit: int = 50,
    ) -> tuple[RankedSymbol, ...]:
        if not documents:
            return ()
        eligible = {item.node_id: item for item in documents}
        adjacency: dict[str, set[str]] = defaultdict(set)
        for edge in graph.edges:
            if edge.relation in {
                "calls",
                "defines",
                "imports",
                "references",
                "runtime_calls",
                "tested_by",
            }:
                adjacency[edge.source].add(edge.target)
                adjacency[edge.target].add(edge.source)
        seeds = {
            item.node_id
            for item in documents
            if item.executed or item.traceback_distance == 0.0
        }
        scores = {node_id: (1.0 / max(1, len(seeds)) if node_id in seeds else 0.0) for node_id in eligible}
        personalization = dict(scores)
        for _ in range(24):
            updated = {
                node_id: 0.15 * personalization.get(node_id, 0.0)
                for node_id in eligible
            }
            for source, score in scores.items():
                targets = [target for target in adjacency[source] if target in eligible]
                if not targets:
                    updated[source] += 0.85 * score
                    continue
                share = 0.85 * score / len(targets)
                for target in targets:
                    updated[target] += share
            scores = updated
        ranked = [
            RankedSymbol(
                item.node_id,
                item.file_path,
                item.symbol,
                scores[item.node_id],
                ("repograph",),
                item.line_count,
                item.obligations,
            )
            for item in documents
            if scores[item.node_id] > 0
        ]
        return tuple(
            sorted(
                ranked,
                key=lambda item: (-item.score, item.file_path, item.symbol or ""),
            )[:limit]
        )


def reciprocal_rank_fusion(
    rankings: Sequence[Sequence[RankedSymbol]],
    *,
    k: int = 60,
    limit: int = 50,
) -> tuple[RankedSymbol, ...]:
    scores: dict[str, float] = defaultdict(float)
    records: dict[str, RankedSymbol] = {}
    sources: dict[str, set[str]] = defaultdict(set)
    for ranking in rankings:
        for rank, item in enumerate(ranking, start=1):
            scores[item.node_id] += 1.0 / (k + rank)
            records[item.node_id] = item
            sources[item.node_id].update(item.rank_sources)
    fused = [
        RankedSymbol(
            node_id,
            records[node_id].file_path,
            records[node_id].symbol,
            score,
            tuple(sorted(sources[node_id])),
            records[node_id].line_count,
            records[node_id].obligations,
            records[node_id].evidence,
        )
        for node_id, score in scores.items()
    ]
    return tuple(
        sorted(
            fused,
            key=lambda item: (-item.score, item.file_path, item.symbol or ""),
        )[:limit]
    )


class StructuralReranker:
    """Fixed feature combiner; learned weights must be supplied by a method lock."""

    DEFAULT_WEIGHTS: ClassVar[dict[str, float]] = {
        "base": 1.0,
        "executed": 0.35,
        "traceback": 0.55,
        "dynamic": 0.25,
        "obligations": 0.20,
        "risk": -0.05,
    }

    def __init__(self, weights: dict[str, float] | None = None) -> None:
        self.weights = dict(weights or self.DEFAULT_WEIGHTS)

    def rank(
        self,
        ranking: Sequence[RankedSymbol],
        documents: Sequence[SymbolDocument],
        *,
        limit: int = 20,
    ) -> tuple[RankedSymbol, ...]:
        by_id = {item.node_id: item for item in documents}
        rescored = []
        for item in ranking:
            document = by_id[item.node_id]
            risk = math.log1p(
                float(document.attributes.get("fan_out", 0))
            )
            score = (
                self.weights["base"] * item.score
                + self.weights["executed"] * float(document.executed)
                + self.weights["traceback"]
                * float(document.traceback_distance == 0.0)
                + self.weights["dynamic"]
                * float(document.dynamic_call_distance is not None)
                + self.weights["obligations"] * len(document.obligations)
                + self.weights["risk"] * risk
            )
            rescored.append(
                RankedSymbol(
                    item.node_id,
                    item.file_path,
                    item.symbol,
                    score,
                    (*item.rank_sources, "structural_reranker"),
                    item.line_count,
                    item.obligations,
                    item.evidence,
                )
            )
        return tuple(
            sorted(
                rescored,
                key=lambda item: (-item.score, item.file_path, item.symbol or ""),
            )[:limit]
        )
