from __future__ import annotations

import hashlib
import math
import re
from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from itertools import pairwise
from typing import ClassVar, Protocol

from .graph import RepositoryGraph
from .runtime_events import RuntimeEvent

TOKEN_RE = re.compile(r"[A-Za-z_][A-Za-z0-9_]{1,}|[0-9]+")
CAMEL_BOUNDARY_RE = re.compile(
    r"(?<=[a-z0-9])(?=[A-Z])|(?<=[A-Z])(?=[A-Z][a-z])"
)
ABSOLUTE_PATH_RE = re.compile(
    r"(?:/home/runner/work|/tmp|/private/tmp|[A-Za-z]:\\Temp)"
    r"[/\\][^\s:]+"
)
SOURCE_KINDS = frozenset({"class", "configuration_key", "dependency", "file", "function", "method"})
R9_SOURCE_KINDS = SOURCE_KINDS | frozenset(
    {"module", "runtime_symbol", "serialized_artifact"}
)
TRACEBACK_KINDS = frozenset({"runtime_exception", "runtime_traceback_frame", "traceback"})
COMMON_IDENTIFIER_TOKENS = frozenset(
    {
        "config",
        "data",
        "get",
        "init",
        "load",
        "main",
        "run",
        "set",
        "test",
        "tests",
        "value",
    }
)
NOISE_MARKERS = (
    "_distutils_hack",
    "distutils-precedence.pth",
    "runtime_launcher.py",
    "site-packages",
    "deprecationwarning",
    "error processing line 1 of",
)


def tokenize(value: str) -> tuple[str, ...]:
    """Tokenize prose and split Python identifiers into searchable subtokens."""
    values: list[str] = []
    for raw in TOKEN_RE.findall(value):
        lowered = raw.lower()
        values.append(lowered)
        parts = re.split(r"[_./\\:-]+", raw)
        for part in parts:
            if not part:
                continue
            for subtoken in CAMEL_BOUNDARY_RE.split(part):
                normalized = subtoken.lower()
                if len(normalized) >= 2 and normalized != lowered:
                    values.append(normalized)
                    if len(normalized) > 4 and normalized.endswith("s"):
                        values.append(normalized[:-1])
    return tuple(values)


@dataclass(frozen=True)
class NormalizedIncident:
    issue: str
    failing_tests: str
    traceback: str
    assertion: str
    exception: str
    weighted_text: str
    identifiers: tuple[str, ...]


class IncidentNormalizer:
    """Remove collection noise and preserve evidence channels separately."""

    CHANNEL_WEIGHTS: ClassVar[dict[str, int]] = {
        "assertion": 4,
        "failing_tests": 3,
        "traceback": 3,
        "exception": 3,
        "issue": 2,
    }

    def normalize(self, query: IncidentQuery) -> NormalizedIncident:
        issue = self._clean(query.issue)
        traceback = self._clean(query.traceback)
        assertion = self._clean(query.assertion)
        failing_tests = " ".join(query.failing_tests)
        exception = self._exception_text(traceback, assertion)
        channels = {
            "issue": issue,
            "failing_tests": failing_tests,
            "traceback": traceback,
            "assertion": assertion,
            "exception": exception,
        }
        weighted = " ".join(
            text
            for name, text in channels.items()
            for _ in range(self.CHANNEL_WEIGHTS[name])
            if text
        )
        identifiers = tuple(
            dict.fromkeys(
                token
                for text in (
                    failing_tests,
                    traceback,
                    assertion,
                    issue,
                )
                for token in tokenize(text)
                if len(token) >= 3 and token not in {"test", "tests", "assert"}
            )
        )
        return NormalizedIncident(
            issue,
            failing_tests,
            traceback,
            assertion,
            exception,
            weighted,
            identifiers,
        )

    @staticmethod
    def _clean(value: str) -> str:
        lines: list[str] = []
        previous = ""
        for raw in value.splitlines():
            lowered = raw.lower()
            if any(marker in lowered for marker in NOISE_MARKERS):
                continue
            line = ABSOLUTE_PATH_RE.sub("<runtime-path>", raw).strip()
            if not line or line == previous:
                continue
            if re.fullmatch(r"[-=_]{3,}", line):
                continue
            lines.append(line)
            previous = line
        return "\n".join(lines[-160:])

    @staticmethod
    def _exception_text(traceback: str, assertion: str) -> str:
        patterns = (
            r"(?m)^([A-Za-z_][A-Za-z0-9_.]*(?:Error|Exception):[^\n]+)$",
            r"(?m)^(AssertionError(?::[^\n]+)?)$",
        )
        matches = [
            match.group(1)
            for source in (assertion, traceback)
            for pattern in patterns
            for match in re.finditer(pattern, source)
        ]
        return " ".join(dict.fromkeys(matches[-8:]))


@dataclass(frozen=True)
class IncidentQuery:
    incident_id: str
    issue: str
    failing_tests: tuple[str, ...]
    traceback: str
    assertion: str = ""

    @property
    def text(self) -> str:
        return IncidentNormalizer().normalize(self).weighted_text


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
    directed_caller_distance: float | None = None
    directed_callee_distance: float | None = None
    execution_frequency: int = 0
    last_touch_proximity: float = 0.0
    failing_test_frequency: float = 0.0


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


def documents_from_graph(
    graph: RepositoryGraph,
    runtime_events: Sequence[RuntimeEvent] = (),
    *,
    source_kinds: frozenset[str] = SOURCE_KINDS,
) -> tuple[SymbolDocument, ...]:
    evidence_by_id = {item.evidence_id: item for item in graph.evidence}
    nodes_by_id = {node.node_id: node for node in graph.nodes}
    executed_nodes = {
        node_id
        for edge in graph.edges
        if edge.relation in {"executes", "runtime_calls", "tested_by"}
        for node_id in (
            (edge.source, edge.target)
            if edge.relation == "runtime_calls"
            else (edge.target,)
        )
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
    runtime_adjacency: dict[str, set[str]] = defaultdict(set)
    forward_adjacency: dict[str, set[str]] = defaultdict(set)
    reverse_adjacency: dict[str, set[str]] = defaultdict(set)
    for edge in graph.edges:
        if edge.relation != "runtime_calls":
            continue
        runtime_adjacency[edge.source].add(edge.target)
        runtime_adjacency[edge.target].add(edge.source)
        forward_adjacency[edge.source].add(edge.target)
        reverse_adjacency[edge.target].add(edge.source)
    runtime_distance = {node_id: 0 for node_id in traceback_nodes}
    frontier = list(traceback_nodes)
    while frontier:
        source = frontier.pop(0)
        for target in runtime_adjacency[source]:
            if target in runtime_distance:
                continue
            runtime_distance[target] = runtime_distance[source] + 1
            frontier.append(target)
    caller_distance = _reverse_distance(traceback_nodes, reverse_adjacency)
    callee_distance = _reverse_distance(traceback_nodes, forward_adjacency)
    runtime_profile = _runtime_profile(graph, runtime_events)
    obligations_by_node: dict[str, set[str]] = defaultdict(set)
    for edge in graph.edges:
        target = nodes_by_id.get(edge.target)
        if target and target.kind == "runtime_exception":
            obligation = target.attributes.get("obligation")
            if obligation:
                obligations_by_node[edge.source].add(str(obligation))
    documents = []
    for node in graph.nodes:
        attributes = dict(node.attributes)
        file_path = node.file_path or str(
            attributes.get("artifact_path")
            or attributes.get("module_path")
            or attributes.get("path")
            or ""
        )
        if node.kind not in source_kinds or not file_path:
            continue
        node_evidence = " ".join(
            evidence_by_id[reference].detail
            for reference in node.evidence_refs
            if reference in evidence_by_id
            and evidence_by_id[reference].kind in TRACEBACK_KINDS
        )[:4096]
        semantic = " ".join(
            str(value)
            for value in node.attributes.get("semantic_tokens", ())
        )
        text = " ".join(
            (
                file_path,
                node.symbol or "",
                node.kind,
                semantic,
                str(node.attributes.get("source_excerpt", "")),
                node_evidence if node.node_id in traceback_nodes else "",
            )
        )
        attributes.setdefault("node_kind", node.kind)
        line_count = max(
            1,
            int(
                attributes.get(
                    "line_count",
                    max(
                        1,
                        int(attributes.get("end_lineno", 1))
                        - int(attributes.get("lineno", 1))
                        + 1,
                    ),
                )
            ),
        )
        profile = runtime_profile.get(node.node_id, {})
        documents.append(
            SymbolDocument(
                node.node_id,
                file_path,
                node.symbol,
                text,
                line_count,
                node.node_id in executed_nodes or node.node_id in traceback_nodes,
                0.0 if node.node_id in traceback_nodes else None,
                (
                    float(runtime_distance[node.node_id])
                    if node.node_id in runtime_distance
                    else None
                ),
                tuple(sorted(obligations_by_node[node.node_id])),
                attributes,
                (
                    float(caller_distance[node.node_id])
                    if node.node_id in caller_distance
                    else None
                ),
                (
                    float(callee_distance[node.node_id])
                    if node.node_id in callee_distance
                    else None
                ),
                int(profile.get("execution_frequency", 0)),
                float(profile.get("last_touch_proximity", 0.0)),
                float(profile.get("failing_test_frequency", 0.0)),
            )
        )
    return tuple(documents)


def _reverse_distance(
    seeds: set[str],
    adjacency: dict[str, set[str]],
) -> dict[str, int]:
    distance = {node_id: 0 for node_id in seeds}
    frontier = list(seeds)
    while frontier:
        source = frontier.pop(0)
        for target in adjacency[source]:
            if target in distance:
                continue
            distance[target] = distance[source] + 1
            frontier.append(target)
    return distance


def _runtime_profile(
    graph: RepositoryGraph,
    runtime_events: Sequence[RuntimeEvent],
) -> dict[str, dict[str, float]]:
    if not runtime_events:
        return {}
    nodes = tuple(graph.nodes)
    event_positions: dict[str, list[int]] = defaultdict(list)
    event_tests: dict[str, set[str]] = defaultdict(set)
    total = len(runtime_events)
    all_tests = {event.test_id for event in runtime_events}

    def node_for(file_path: str | None, symbol: str | None) -> str | None:
        if not file_path:
            return None
        candidates = [
            node
            for node in nodes
            if node.file_path == file_path
            and (
                symbol is None
                or node.symbol == symbol
                or (node.symbol or "").rsplit(".", 1)[-1] == symbol
            )
        ]
        symbolic = next((node for node in candidates if node.symbol), None)
        node = symbolic or next(iter(candidates), None)
        return node.node_id if node is not None else None

    for index, event in enumerate(runtime_events):
        identifiers = {
            node_for(event.source_file, event.source_symbol),
            node_for(event.target_file, event.target_symbol),
        }
        for node_id in identifiers - {None}:
            event_positions[node_id].append(index)
            event_tests[node_id].add(event.test_id)
    return {
        node_id: {
            "execution_frequency": float(len(positions)),
            "last_touch_proximity": 1.0
            / (1.0 + max(0, total - 1 - max(positions))),
            "failing_test_frequency": len(event_tests[node_id])
            / max(1, len(all_tests)),
        }
        for node_id, positions in event_positions.items()
    }


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
    weights: Sequence[float] | None = None,
) -> tuple[RankedSymbol, ...]:
    if weights is not None and len(weights) != len(rankings):
        raise ValueError("RRF weights must match the number of rankings")
    scores: dict[str, float] = defaultdict(float)
    records: dict[str, RankedSymbol] = {}
    sources: dict[str, set[str]] = defaultdict(set)
    for ranking_index, ranking in enumerate(rankings):
        weight = weights[ranking_index] if weights is not None else 1.0
        for rank, item in enumerate(ranking, start=1):
            scores[item.node_id] += weight / (k + rank)
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


class ExactSymbolExtractor:
    """Keep exact identifier evidence even when rank fusion is diffuse."""

    def rank(
        self,
        query: IncidentQuery,
        documents: Sequence[SymbolDocument],
    ) -> tuple[RankedSymbol, ...]:
        normalized = IncidentNormalizer().normalize(query)
        query_tokens = set(normalized.identifiers)
        values = []
        for document in documents:
            symbol_tokens = set(tokenize(document.symbol or ""))
            file_tokens = set(tokenize(document.file_path))
            overlap = symbol_tokens.intersection(query_tokens)
            semantic_tokens = set(
                tokenize(
                    " ".join(
                        str(item)
                        for item in document.attributes.get(
                            "semantic_tokens",
                            (),
                        )
                    )
                )
            )
            semantic_overlap = semantic_tokens.intersection(query_tokens)
            exact_symbol = bool(
                document.symbol
                and (document.symbol.lower() in normalized.weighted_text.lower())
            )
            if (
                not overlap
                and not exact_symbol
                and len(semantic_overlap) < 2
            ):
                continue
            score = (
                4.0 * float(exact_symbol)
                + 1.5 * len(overlap)
                + 0.65 * len(semantic_overlap)
                + 0.25 * len(file_tokens.intersection(query_tokens))
            )
            values.append(
                RankedSymbol(
                    document.node_id,
                    document.file_path,
                    document.symbol,
                    score,
                    ("exact_symbol",),
                    document.line_count,
                    document.obligations,
                    tuple(
                        sorted(
                            (
                                *(
                                    f"identifier:{item}"
                                    for item in overlap
                                ),
                                *(
                                    f"semantic_identifier:{item}"
                                    for item in semantic_overlap
                                ),
                            )
                        )
                    ),
                )
            )
        return tuple(
            sorted(
                values,
                key=lambda item: (-item.score, item.file_path, item.symbol or ""),
            )
        )


class StrictIdentifierExtractor:
    """Return literal or rare identifier matches, never semantic-token matches."""

    MAXIMUM_CANDIDATES = 500

    def rank(
        self,
        query: IncidentQuery,
        documents: Sequence[SymbolDocument],
        *,
        limit: int = MAXIMUM_CANDIDATES,
    ) -> tuple[RankedSymbol, ...]:
        normalized = IncidentNormalizer().normalize(query)
        query_text = normalized.weighted_text.lower()
        query_tokens = set(normalized.identifiers) - COMMON_IDENTIFIER_TOKENS
        document_tokens = [
            set(tokenize(document.symbol or ""))
            | set(tokenize(document.file_path))
            for document in documents
        ]
        frequencies = Counter(
            token for tokens in document_tokens for token in tokens
        )
        rare_limit = max(2, math.ceil(len(documents) * 0.01))
        values = []
        for document, tokens in zip(documents, document_tokens):
            symbol = (document.symbol or "").lower()
            terminal = symbol.rsplit(".", 1)[-1]
            file_stem = document.file_path.rsplit("/", 1)[-1].split(".", 1)[0]
            full_match = bool(symbol and symbol in query_text)
            terminal_match = bool(
                terminal
                and terminal not in COMMON_IDENTIFIER_TOKENS
                and re.search(rf"\b{re.escape(terminal)}\b", query_text)
            )
            file_match = bool(
                file_stem
                and file_stem not in COMMON_IDENTIFIER_TOKENS
                and re.search(rf"\b{re.escape(file_stem.lower())}\b", query_text)
            )
            rare = sorted(
                token
                for token in tokens.intersection(query_tokens)
                if len(token) >= 4 and frequencies[token] <= rare_limit
            )
            if not full_match and not terminal_match and not file_match and not rare:
                continue
            level = (
                "full_symbol"
                if full_match
                else "terminal_symbol"
                if terminal_match
                else "file_basename"
                if file_match
                else "rare_identifier"
            )
            score = (
                4.0 * float(full_match)
                + 3.0 * float(terminal_match)
                + 2.0 * float(file_match)
                + min(len(rare), 3)
            )
            values.append(
                RankedSymbol(
                    document.node_id,
                    document.file_path,
                    document.symbol,
                    score,
                    ("strict_identifier",),
                    document.line_count,
                    document.obligations,
                    (
                        f"exact_identifier_level:{level}",
                        *(f"rare_identifier:{token}" for token in rare),
                    ),
                )
            )
        bounded = max(1, min(limit, self.MAXIMUM_CANDIDATES))
        return tuple(
            sorted(
                values,
                key=lambda item: (-item.score, item.file_path, item.symbol or ""),
            )[:bounded]
        )


class CandidateReservoir:
    """Union independent channels before the final top-k decision."""

    MINIMUM_LIMIT = 150
    MAXIMUM_LIMIT = 300

    def build(
        self,
        channels: Sequence[Sequence[RankedSymbol]],
        *,
        limit: int = MAXIMUM_LIMIT,
        weights: Sequence[float] | None = None,
    ) -> tuple[RankedSymbol, ...]:
        bounded = max(self.MINIMUM_LIMIT, min(limit, self.MAXIMUM_LIMIT))
        if weights is not None and len(weights) != len(channels):
            raise ValueError(
                "reservoir weights must match the number of channels"
            )
        scores: dict[str, float] = defaultdict(float)
        records: dict[str, RankedSymbol] = {}
        sources: dict[str, set[str]] = defaultdict(set)
        evidence: dict[str, set[str]] = defaultdict(set)
        for channel_index, ranking in enumerate(channels):
            weight = weights[channel_index] if weights is not None else 1.0
            channel_name = (
                ranking[0].rank_sources[0] if ranking else f"channel-{channel_index}"
            )
            for rank, item in enumerate(ranking[:bounded], start=1):
                scores[item.node_id] += weight / math.sqrt(rank)
                records[item.node_id] = item
                sources[item.node_id].update(item.rank_sources)
                evidence[item.node_id].update(item.evidence)
                evidence[item.node_id].add(
                    f"channel_rank:{channel_name}:{rank}"
                )
        values = (
            RankedSymbol(
                node_id,
                item.file_path,
                item.symbol,
                scores[node_id],
                tuple(sorted(sources[node_id])),
                item.line_count,
                item.obligations,
                tuple(sorted(evidence[node_id])),
            )
            for node_id, item in records.items()
        )
        return tuple(
            sorted(
                values,
                key=lambda item: (-item.score, item.file_path, item.symbol or ""),
            )[:bounded]
        )


@dataclass(frozen=True)
class R9CompressionConfig:
    bm25_limit: int = 300
    graph_limit: int = 300
    runtime_limit: int = 300
    legacy_limit: int = 50
    dense_limit: int = 300
    strict_identifier_limit: int = 500
    file_limit: int = 15
    symbols_per_file: int = 3
    rerank_limit: int = 40
    final_limit: int = 20


class R9CandidateCompressor:
    """Compress explicit channel rankings without mixing incomparable raw scores."""

    CHANNEL_WEIGHTS: ClassVar[dict[str, float]] = {
        "bm25": 1.15,
        "runtime": 1.25,
        "repograph": 0.95,
        "strict_identifier": 1.35,
        "legacy": 1.05,
        "dense": 1.00,
        "cross": 1.10,
    }
    CHANNEL_QUOTAS: ClassVar[dict[str, int]] = {
        "bm25": 5,
        "runtime": 4,
        "repograph": 3,
        "strict_identifier": 3,
    }
    DEFAULT_FEATURE_WEIGHTS: ClassVar[dict[str, float]] = {
        **{
            f"rank_{channel}": weight
            for channel, weight in CHANNEL_WEIGHTS.items()
        },
        "channel_consensus": 0.16,
        "best_rank_signal": 0.20,
        "executed": 0.10,
        "traceback": 0.90,
        "dynamic": 0.40,
        "caller": 0.35,
        "callee": 0.15,
        "frequency": -0.03,
        "last_touch": 0.90,
        "failing_test_frequency": 0.60,
    }

    def __init__(
        self,
        config: R9CompressionConfig | None = None,
        *,
        feature_weights: dict[str, float] | None = None,
    ) -> None:
        self.config = config or R9CompressionConfig()
        self.feature_weights = {
            **self.DEFAULT_FEATURE_WEIGHTS,
            **(feature_weights or {}),
        }

    @staticmethod
    def _rank_signal(rank: int) -> float:
        return 1.0 / math.log2(2.0 + rank)

    def rank(
        self,
        channels: dict[str, Sequence[RankedSymbol]],
        documents: Sequence[SymbolDocument],
        *,
        hierarchical: bool,
        limit: int | None = None,
        feature_weights: Mapping[str, float] | None = None,
        feature_rows: Mapping[str, Mapping[str, float]] | None = None,
        score_overrides: Mapping[str, float] | None = None,
    ) -> tuple[RankedSymbol, ...]:
        final_limit = limit or self.config.final_limit
        by_id = {item.node_id: item for item in documents}
        records: dict[str, RankedSymbol] = {}
        sources: dict[str, set[str]] = defaultdict(set)
        evidence: dict[str, set[str]] = defaultdict(set)
        for channel, ranking in channels.items():
            for rank, item in enumerate(ranking, start=1):
                if item.node_id not in by_id:
                    continue
                records[item.node_id] = item
                sources[item.node_id].update(item.rank_sources)
                evidence[item.node_id].update(item.evidence)
                evidence[item.node_id].add(f"r9_channel_rank:{channel}:{rank}")

        features = feature_rows or self.feature_rows(channels, documents)
        weights = feature_weights or self.feature_weights
        scored: dict[str, RankedSymbol] = {}
        for node_id, item in records.items():
            score = (
                score_overrides[node_id]
                if score_overrides is not None
                else sum(
                    weights.get(name, 0.0) * value
                    for name, value in features[node_id].items()
                )
            )
            scored[node_id] = RankedSymbol(
                item.node_id,
                item.file_path,
                item.symbol,
                score,
                tuple(sorted((*sources[node_id], "r9_normalized_rank"))),
                item.line_count,
                item.obligations,
                tuple(sorted(evidence[node_id])),
            )

        ordered = sorted(
            scored.values(),
            key=lambda item: (-item.score, item.file_path, item.symbol or ""),
        )
        eligible = self._hierarchical_pool(ordered) if hierarchical else ordered
        eligible_ids = {item.node_id for item in eligible}
        protected_ids = {
            item.node_id
            for channel, quota in self.CHANNEL_QUOTAS.items()
            for item in channels.get(channel, ())[:quota]
        }
        eligible_ids.update(protected_ids)
        eligible = [item for item in ordered if item.node_id in eligible_ids]

        selected: list[RankedSymbol] = []
        selected_ids: set[str] = set()
        for channel, quota in self.CHANNEL_QUOTAS.items():
            for item in channels.get(channel, ())[:quota]:
                candidate = scored.get(item.node_id)
                if candidate is None or candidate.node_id in selected_ids:
                    continue
                selected.append(candidate)
                selected_ids.add(candidate.node_id)
                if len(selected) >= final_limit:
                    break
            if len(selected) >= final_limit:
                break
        for item in eligible:
            if item.node_id in selected_ids:
                continue
            selected.append(item)
            selected_ids.add(item.node_id)
            if len(selected) >= final_limit:
                break
        return tuple(
            sorted(
                selected,
                key=lambda item: (-item.score, item.file_path, item.symbol or ""),
            )
        )

    def feature_rows(
        self,
        channels: Mapping[str, Sequence[RankedSymbol]],
        documents: Sequence[SymbolDocument],
    ) -> dict[str, dict[str, float]]:
        """Return only observable, candidate-specific compression features."""
        by_id = {item.node_id: item for item in documents}
        ranks: dict[str, dict[str, int]] = defaultdict(dict)
        for channel, ranking in channels.items():
            for rank, item in enumerate(ranking, start=1):
                if item.node_id in by_id:
                    ranks[item.node_id][channel] = rank

        rows: dict[str, dict[str, float]] = {}
        for node_id, channel_ranks in ranks.items():
            document = by_id[node_id]
            rank_signals = {
                channel: self._rank_signal(rank)
                for channel, rank in channel_ranks.items()
            }
            rows[node_id] = {
                **{
                    f"rank_{channel}": rank_signals.get(channel, 0.0)
                    for channel in self.CHANNEL_WEIGHTS
                },
                "channel_consensus": len(channel_ranks)
                / max(1.0, float(len(channels))),
                "best_rank_signal": max(rank_signals.values(), default=0.0),
                "executed": float(document.executed),
                "traceback": float(document.traceback_distance == 0.0),
                "dynamic": self._distance_signal(
                    document.dynamic_call_distance
                ),
                "caller": self._distance_signal(
                    document.directed_caller_distance
                ),
                "callee": self._distance_signal(
                    document.directed_callee_distance
                ),
                "frequency": math.log1p(document.execution_frequency),
                "last_touch": document.last_touch_proximity,
                "failing_test_frequency": document.failing_test_frequency,
            }
        return rows

    def _hierarchical_pool(
        self,
        ordered: Sequence[RankedSymbol],
    ) -> list[RankedSymbol]:
        by_file: dict[str, list[RankedSymbol]] = defaultdict(list)
        for item in ordered:
            by_file[item.file_path].append(item)
        file_scores = {
            file_path: values[0].score
            + 0.20 * sum(item.score for item in values[1:3])
            for file_path, values in by_file.items()
        }
        selected_files = {
            file_path
            for file_path, _ in sorted(
                file_scores.items(),
                key=lambda value: (-value[1], value[0]),
            )[: self.config.file_limit]
        }
        values = [
            item
            for file_path in selected_files
            for item in by_file[file_path][: self.config.symbols_per_file]
        ]
        return sorted(
            values,
            key=lambda item: (-item.score, item.file_path, item.symbol or ""),
        )[: self.config.rerank_limit]

    @staticmethod
    def _distance_signal(distance: float | None) -> float:
        return 1.0 / (1.0 + distance) if distance is not None else 0.0


class StructuralReranker:
    """Fixed feature combiner; learned weights must be supplied by a method lock."""

    DEFAULT_WEIGHTS: ClassVar[dict[str, float]] = {
        "base": 1.0,
        "executed": 0.35,
        "traceback": 0.55,
        "dynamic": 0.25,
        "caller": 0.22,
        "callee": 0.10,
        "frequency": 0.08,
        "last_touch": 0.20,
        "failing_test_frequency": 0.25,
        "exact_symbol": 0.55,
        "assertion_overlap": 0.30,
        "obligations": 0.20,
        "risk": -0.05,
        "test_path": -1.00,
        "documentation_path": -0.35,
    }

    def __init__(self, weights: dict[str, float] | None = None) -> None:
        self.weights = dict(weights or self.DEFAULT_WEIGHTS)

    def rank(
        self,
        ranking: Sequence[RankedSymbol],
        documents: Sequence[SymbolDocument],
        *,
        limit: int = 20,
        query: IncidentQuery | None = None,
    ) -> tuple[RankedSymbol, ...]:
        by_id = {item.node_id: item for item in documents}
        normalized = (
            IncidentNormalizer().normalize(query) if query is not None else None
        )
        query_identifiers = set(normalized.identifiers) if normalized else set()
        assertion_tokens = (
            set(tokenize(normalized.assertion)) if normalized else set()
        )
        rescored = []
        for item in ranking:
            document = by_id[item.node_id]
            risk = math.log1p(
                float(document.attributes.get("fan_out", 0))
            )
            symbol_tokens = set(tokenize(document.symbol or ""))
            document_tokens = set(tokenize(document.text))
            exact_symbol = float(
                bool(symbol_tokens.intersection(query_identifiers))
                or "exact_symbol" in item.rank_sources
            )
            assertion_overlap = (
                len(assertion_tokens.intersection(document_tokens))
                / max(1, len(assertion_tokens))
            )
            normalized_path = document.file_path.lower().replace("\\", "/")
            is_test = (
                normalized_path.startswith(("test/", "tests/"))
                or "/test/" in normalized_path
                or "/tests/" in normalized_path
                or normalized_path.endswith("_test.py")
            )
            is_documentation = normalized_path.startswith(
                ("doc/", "docs/", "example/", "examples/")
            )
            score = (
                self.weights["base"] * item.score
                + self.weights["executed"] * float(document.executed)
                + self.weights["traceback"]
                * float(document.traceback_distance == 0.0)
                + self.weights["dynamic"]
                * (
                    1.0 / (1.0 + document.dynamic_call_distance)
                    if document.dynamic_call_distance is not None
                    else 0.0
                )
                + self.weights["caller"]
                * (
                    1.0 / (1.0 + document.directed_caller_distance)
                    if document.directed_caller_distance is not None
                    else 0.0
                )
                + self.weights["callee"]
                * (
                    1.0 / (1.0 + document.directed_callee_distance)
                    if document.directed_callee_distance is not None
                    else 0.0
                )
                + self.weights["frequency"]
                * math.log1p(document.execution_frequency)
                + self.weights["last_touch"] * document.last_touch_proximity
                + self.weights["failing_test_frequency"]
                * document.failing_test_frequency
                + self.weights["exact_symbol"] * exact_symbol
                + self.weights["assertion_overlap"] * assertion_overlap
                + self.weights["obligations"] * len(document.obligations)
                + self.weights["risk"] * risk
                + self.weights["test_path"] * float(is_test)
                + self.weights["documentation_path"] * float(is_documentation)
            )
            evidence = (
                *item.evidence,
                f"execution_frequency:{document.execution_frequency}",
                f"last_touch_proximity:{document.last_touch_proximity:.6f}",
                f"assertion_overlap:{assertion_overlap:.6f}",
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
                    evidence,
                )
            )
        return tuple(
            sorted(
                rescored,
                key=lambda item: (-item.score, item.file_path, item.symbol or ""),
            )[:limit]
        )
