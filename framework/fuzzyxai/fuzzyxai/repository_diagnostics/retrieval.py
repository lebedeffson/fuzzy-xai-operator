from __future__ import annotations

import math
from collections import defaultdict, deque
from dataclasses import dataclass, field

from .graph import EvidenceRef, RepositoryEdge, RepositoryGraph, RepositoryNode

NON_REPAIRABLE_KINDS = frozenset(
    {
        "fixture",
        "module",
        "package",
        "repository",
        "runtime_exception",
        "runtime_test_support",
        "source_parse_state",
        "test",
    }
)


@dataclass(frozen=True)
class RetrievalSignal:
    kind: str
    score: float
    obligation: str
    evidence_refs: tuple[str, ...]


@dataclass(frozen=True)
class CandidateFeatures:
    traceback_distance: float | None = None
    dynamic_call_distance: float | None = None
    executed_in_failing_path: bool = False
    assertion_value_owner: bool = False
    artifact_access_match: bool = False
    configuration_reference_match: bool = False
    dependency_constraint_match: bool = False
    test_symbol_proximity: float = 0.0
    fan_out: int = 0
    fan_in: int = 0
    historical_change_frequency: float = 0.0


@dataclass(frozen=True)
class RetrievedCandidate:
    node_id: str
    repository: str
    file_path: str | None
    symbol: str | None
    retrieval_score: float
    confidence: float
    covered_obligations: tuple[str, ...]
    evidence_refs: tuple[str, ...]
    signals: tuple[RetrievalSignal, ...]
    features: CandidateFeatures = field(default_factory=CandidateFeatures)


def _is_test_path(path: str | None) -> bool:
    normalized = (path or "").replace("\\", "/").lower()
    return normalized.startswith(("test/", "tests/")) or "/tests/" in normalized or "/test_" in normalized or normalized.endswith("_test.py")


def _eligible(node: RepositoryNode) -> bool:
    return node.kind not in NON_REPAIRABLE_KINDS and not _is_test_path(node.file_path)


class EvidenceGroundedCandidateRetriever:
    """Retrieve candidates only when observable runtime structure reaches them."""

    def __init__(
        self,
        *,
        max_call_depth: int = 6,
        max_candidates: int = 20,
    ) -> None:
        self.max_call_depth = max_call_depth
        self.max_candidates = max_candidates

    def retrieve(self, graph: RepositoryGraph) -> tuple[RetrievedCandidate, ...]:
        nodes = {node.node_id: node for node in graph.nodes}
        evidence = {item.evidence_id: item for item in graph.evidence}
        signals: dict[str, list[RetrievalSignal]] = defaultdict(list)
        runtime_obligation = {
            node.node_id: str(node.attributes["obligation"]) for node in graph.nodes if node.kind == "runtime_exception" and node.attributes.get("obligation")
        }
        failing_test_obligation = {
            edge.source: runtime_obligation[edge.target] for edge in graph.edges if edge.relation == "fails_in" and edge.target in runtime_obligation
        }

        self._traceback_signals(
            graph.edges,
            nodes,
            evidence,
            runtime_obligation,
            signals,
        )
        self._traceback_neighborhood_signals(
            graph.edges,
            nodes,
            signals,
        )
        self._test_call_signals(
            graph.edges,
            nodes,
            failing_test_obligation,
            signals,
        )
        self._dynamic_execution_signals(
            graph.edges,
            nodes,
            failing_test_obligation,
            signals,
        )
        self._dynamic_coverage_signals(graph.evidence, nodes, graph.obligations, signals)
        self._runtime_io_signals(graph.edges, nodes, signals)
        self._runtime_context_signals(graph.edges, nodes, signals)
        self._manifest_reference_signals(graph, nodes, signals)
        self._runtime_semantic_boost(graph, nodes, signals)

        candidates = []
        for node_id, values in signals.items():
            node = nodes.get(node_id)
            if node is None or not _eligible(node):
                continue
            merged = self._merge_signals(values)
            score = sum(item.score for item in merged)
            if score <= 0:
                continue
            obligations = tuple(sorted({item.obligation for item in merged if item.obligation}))
            if not obligations:
                continue
            refs = tuple(
                sorted(
                    {
                        *node.evidence_refs,
                        *(ref for item in merged for ref in item.evidence_refs),
                    }
                )
            )
            candidates.append(
                RetrievedCandidate(
                    node.node_id,
                    node.repository,
                    node.file_path,
                    node.symbol,
                    score,
                    1.0 - math.exp(-score / 12.0),
                    obligations,
                    refs,
                    merged,
                    self._features(node.node_id, merged, graph),
                )
            )
        return tuple(
            sorted(
                candidates,
                key=lambda item: (
                    -item.retrieval_score,
                    -len(item.covered_obligations),
                    item.file_path or "",
                    item.symbol or "",
                    item.node_id,
                ),
            )[: self.max_candidates]
        )

    @staticmethod
    def _features(
        node_id: str,
        signals: tuple[RetrievalSignal, ...],
        graph: RepositoryGraph,
    ) -> CandidateFeatures:
        kinds = {signal.kind for signal in signals}
        neighborhood_scores = [signal.score for signal in signals if signal.kind == "traceback_call_neighborhood"]
        dynamic_scores = [signal.score for signal in signals if signal.kind == "executed_path_call_graph"]
        fan_out = len({edge.target for edge in graph.edges if edge.source == node_id})
        fan_in = len({edge.source for edge in graph.edges if edge.target == node_id})
        return CandidateFeatures(
            traceback_distance=(
                0.0
                if "exact_traceback_frame" in kinds
                else min(
                    (6.0 / score for score in neighborhood_scores),
                    default=None,
                )
            ),
            dynamic_call_distance=min(
                (11.0 / score for score in dynamic_scores),
                default=None,
            ),
            executed_in_failing_path=bool(
                kinds
                & {
                    "dynamic_failing_test_coverage",
                    "executed_path_call_graph",
                    "exact_traceback_frame",
                }
            ),
            assertion_value_owner=("runtime_assertion_owner" in kinds),
            artifact_access_match="runtime_read_write" in kinds,
            configuration_reference_match=bool(
                kinds
                & {
                    "manifest_configuration_reference",
                    "runtime_configuration_or_dependency",
                }
            ),
            dependency_constraint_match=("runtime_configuration_or_dependency" in kinds),
            test_symbol_proximity=max(
                (
                    signal.score
                    for signal in signals
                    if signal.kind
                    in {
                        "executed_path_call_graph",
                        "test_to_invoked_symbol",
                    }
                ),
                default=0.0,
            ),
            fan_out=fan_out,
            fan_in=fan_in,
            # Historical signals are forbidden until a pre-fix history source exists.
            historical_change_frequency=0.0,
        )

    @staticmethod
    def _merge_signals(values: list[RetrievalSignal]) -> tuple[RetrievalSignal, ...]:
        merged: dict[tuple[str, str], RetrievalSignal] = {}
        for value in values:
            key = (value.kind, value.obligation)
            previous = merged.get(key)
            if previous is None or value.score > previous.score:
                merged[key] = value
            elif value.score == previous.score:
                merged[key] = RetrievalSignal(
                    value.kind,
                    value.score,
                    value.obligation,
                    tuple(sorted({*previous.evidence_refs, *value.evidence_refs})),
                )
        return tuple(
            sorted(
                merged.values(),
                key=lambda item: (item.kind, item.obligation, item.evidence_refs),
            )
        )

    @staticmethod
    def _traceback_signals(
        edges: tuple[RepositoryEdge, ...],
        nodes: dict[str, RepositoryNode],
        evidence: dict[str, EvidenceRef],
        runtime_obligation: dict[str, str],
        signals: dict[str, list[RetrievalSignal]],
    ) -> None:
        for edge in edges:
            if edge.target not in runtime_obligation or edge.relation != "produces":
                continue
            refs = tuple(ref for ref in edge.evidence_refs if evidence.get(ref) is not None and evidence[ref].kind == "traceback")
            if not refs or edge.source not in nodes:
                continue
            signals[edge.source].append(
                RetrievalSignal(
                    "exact_traceback_frame",
                    14.0,
                    runtime_obligation[edge.target],
                    refs,
                )
            )

    def _test_call_signals(
        self,
        edges: tuple[RepositoryEdge, ...],
        nodes: dict[str, RepositoryNode],
        failing_test_obligation: dict[str, str],
        signals: dict[str, list[RetrievalSignal]],
    ) -> None:
        calls: dict[str, list[RepositoryEdge]] = defaultdict(list)
        for edge in edges:
            if edge.relation in {"calls", "runtime_calls"}:
                calls[edge.source].append(edge)
        for test_id, obligation in failing_test_obligation.items():
            queue: deque[tuple[str, int, tuple[str, ...]]] = deque([(test_id, 0, ())])
            visited = {test_id}
            while queue:
                current, depth, path_refs = queue.popleft()
                if depth >= self.max_call_depth:
                    continue
                for edge in calls.get(current, ()):
                    refs = tuple(sorted({*path_refs, *edge.evidence_refs}))
                    target = nodes.get(edge.target)
                    if target is not None and _eligible(target):
                        signals[target.node_id].append(
                            RetrievalSignal(
                                ("executed_path_call_graph" if edge.relation == "runtime_calls" else "test_to_invoked_symbol"),
                                (11.0 / (depth + 1) if edge.relation == "runtime_calls" else 9.0 / (depth + 1)),
                                obligation,
                                refs,
                            )
                        )
                    if edge.target not in visited:
                        visited.add(edge.target)
                        queue.append((edge.target, depth + 1, refs))

    @staticmethod
    def _traceback_neighborhood_signals(
        edges: tuple[RepositoryEdge, ...],
        nodes: dict[str, RepositoryNode],
        signals: dict[str, list[RetrievalSignal]],
    ) -> None:
        adjacency: dict[str, set[str]] = defaultdict(set)
        for edge in edges:
            if edge.relation not in {"calls", "runtime_calls"}:
                continue
            adjacency[edge.source].add(edge.target)
            adjacency[edge.target].add(edge.source)
        exact = {node_id: tuple(signal for signal in values if signal.kind == "exact_traceback_frame") for node_id, values in signals.items()}
        for start, start_signals in exact.items():
            queue = deque([(start, 0)])
            visited = {start}
            while queue:
                current, depth = queue.popleft()
                if depth >= 2:
                    continue
                for neighbor in adjacency.get(current, ()):
                    if neighbor in visited:
                        continue
                    visited.add(neighbor)
                    target = nodes.get(neighbor)
                    next_depth = depth + 1
                    if target is not None and _eligible(target):
                        for source_signal in start_signals:
                            signals[neighbor].append(
                                RetrievalSignal(
                                    "traceback_call_neighborhood",
                                    6.0 / next_depth,
                                    source_signal.obligation,
                                    source_signal.evidence_refs,
                                )
                            )
                    queue.append((neighbor, next_depth))

    @staticmethod
    def _dynamic_execution_signals(
        edges: tuple[RepositoryEdge, ...],
        nodes: dict[str, RepositoryNode],
        failing_test_obligation: dict[str, str],
        signals: dict[str, list[RetrievalSignal]],
    ) -> None:
        for edge in edges:
            if edge.relation != "executes" or edge.source not in failing_test_obligation:
                continue
            target = nodes.get(edge.target)
            if target is None or not _eligible(target):
                continue
            signals[target.node_id].append(
                RetrievalSignal(
                    "dynamic_failing_test_coverage",
                    12.0,
                    failing_test_obligation[edge.source],
                    edge.evidence_refs,
                )
            )

    @staticmethod
    def _dynamic_coverage_signals(
        evidence: tuple[EvidenceRef, ...],
        nodes: dict[str, RepositoryNode],
        obligations: tuple[str, ...],
        signals: dict[str, list[RetrievalSignal]],
    ) -> None:
        for item in evidence:
            if item.kind != "dynamic_coverage":
                continue
            source = item.source.replace("\\", "/")
            detail = item.detail.lower()
            for node in nodes.values():
                if not _eligible(node) or node.file_path != source:
                    continue
                if node.symbol and node.symbol.rsplit(".", 1)[-1].lower() not in detail:
                    continue
                for obligation in obligations:
                    signals[node.node_id].append(
                        RetrievalSignal(
                            "dynamic_failing_test_coverage",
                            12.0,
                            obligation,
                            (item.evidence_id,),
                        )
                    )

    @staticmethod
    def _runtime_io_signals(
        edges: tuple[RepositoryEdge, ...],
        nodes: dict[str, RepositoryNode],
        signals: dict[str, list[RetrievalSignal]],
    ) -> None:
        structural = {
            node_id: tuple(values)
            for node_id, values in signals.items()
            if any(
                value.kind
                in {
                    "dynamic_failing_test_coverage",
                    "exact_traceback_frame",
                    "test_to_invoked_symbol",
                }
                for value in values
            )
        }
        for edge in edges:
            if edge.relation not in {"loads", "reads", "serializes", "writes"}:
                continue
            source_signals = structural.get(edge.source, ())
            target = nodes.get(edge.target)
            if not source_signals or target is None or not _eligible(target):
                continue
            for source_signal in source_signals:
                signals[target.node_id].append(
                    RetrievalSignal(
                        "runtime_read_write",
                        5.0,
                        source_signal.obligation,
                        tuple(
                            sorted(
                                {
                                    *source_signal.evidence_refs,
                                    *edge.evidence_refs,
                                }
                            )
                        ),
                    )
                )

    @staticmethod
    def _runtime_context_signals(
        edges: tuple[RepositoryEdge, ...],
        nodes: dict[str, RepositoryNode],
        signals: dict[str, list[RetrievalSignal]],
    ) -> None:
        structural = {
            node_id: tuple(values)
            for node_id, values in signals.items()
            if any(
                value.kind
                in {
                    "dynamic_failing_test_coverage",
                    "exact_traceback_frame",
                    "test_to_invoked_symbol",
                }
                for value in values
            )
        }
        for edge in edges:
            if edge.relation not in {"configured_by", "depends_on"}:
                continue
            source_signals = structural.get(edge.source, ())
            target = nodes.get(edge.target)
            if not source_signals or target is None or not _eligible(target):
                continue
            for source_signal in source_signals:
                signals[target.node_id].append(
                    RetrievalSignal(
                        "runtime_configuration_or_dependency",
                        7.0,
                        source_signal.obligation,
                        tuple(
                            sorted(
                                {
                                    *source_signal.evidence_refs,
                                    *edge.evidence_refs,
                                }
                            )
                        ),
                    )
                )

    @staticmethod
    def _manifest_reference_signals(
        graph: RepositoryGraph,
        nodes: dict[str, RepositoryNode],
        signals: dict[str, list[RetrievalSignal]],
    ) -> None:
        runtime_text = " ".join(item.detail.lower() for item in graph.evidence if item.kind in {"failing_test", "traceback"})
        for node in nodes.values():
            if node.kind not in {"configuration_key", "dependency"} or not node.symbol:
                continue
            symbol = node.symbol.lower()
            if symbol not in runtime_text:
                continue
            for obligation in graph.obligations:
                signals[node.node_id].append(
                    RetrievalSignal(
                        "manifest_configuration_reference",
                        8.0,
                        obligation,
                        node.evidence_refs,
                    )
                )

    @staticmethod
    def _runtime_semantic_boost(
        graph: RepositoryGraph,
        nodes: dict[str, RepositoryNode],
        signals: dict[str, list[RetrievalSignal]],
    ) -> None:
        runtime_text = " ".join(item.detail.lower() for item in graph.evidence if item.kind in {"failing_test", "traceback"})
        for node_id in tuple(signals):
            node = nodes[node_id]
            tokens = {str(value).lower() for value in node.attributes.get("semantic_tokens", ())}
            overlap = {token for token in tokens if len(token) >= 4 and token in runtime_text}
            if not overlap:
                continue
            for obligation in {signal.obligation for signal in signals[node_id] if signal.obligation}:
                signals[node_id].append(
                    RetrievalSignal(
                        "runtime_semantic_corroboration",
                        min(2.0, 0.25 * len(overlap)),
                        obligation,
                        (),
                    )
                )
