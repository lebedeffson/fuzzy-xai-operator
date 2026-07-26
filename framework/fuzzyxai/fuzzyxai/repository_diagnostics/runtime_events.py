from __future__ import annotations

import hashlib
import json
from collections.abc import Iterable
from dataclasses import dataclass, replace
from pathlib import Path

from .graph import EvidenceRef, RepositoryEdge, RepositoryGraph, RepositoryNode
from .importer import FORBIDDEN_GOLD_FIELDS

EVENT_KINDS = frozenset(
    {
        "assertion",
        "call",
        "config_read",
        "coverage",
        "dependency",
        "exception",
        "import",
        "read",
        "traceback_frame",
        "write",
    }
)


@dataclass(frozen=True)
class RuntimeEvent:
    event_id: str
    test_id: str
    kind: str
    source_file: str
    source_symbol: str | None = None
    target_file: str | None = None
    target_symbol: str | None = None
    detail: str = ""

    @classmethod
    def from_mapping(cls, value: dict[str, object]) -> RuntimeEvent:
        leaked = FORBIDDEN_GOLD_FIELDS.intersection(value)
        if leaked:
            raise ValueError(f"Gold fields are forbidden in runtime events: {sorted(leaked)}")
        kind = str(value.get("kind", ""))
        if kind not in EVENT_KINDS:
            raise ValueError(f"unsupported runtime event kind: {kind}")
        event = cls(
            event_id=str(value.get("event_id", "")),
            test_id=str(value.get("test_id", "")),
            kind=kind,
            source_file=str(value.get("source_file", "")).replace("\\", "/"),
            source_symbol=_optional_text(value.get("source_symbol")),
            target_file=_optional_text(value.get("target_file")),
            target_symbol=_optional_text(value.get("target_symbol")),
            detail=str(value.get("detail", "")),
        )
        if not event.event_id or not event.test_id or not event.source_file:
            raise ValueError("runtime event requires event_id, test_id and source_file")
        return event


def _optional_text(value: object) -> str | None:
    text = str(value or "").strip()
    return text.replace("\\", "/") if text else None


def load_runtime_events(path: Path) -> tuple[RuntimeEvent, ...]:
    events = tuple(RuntimeEvent.from_mapping(json.loads(line)) for line in path.read_text(encoding="utf-8").splitlines() if line.strip())
    identifiers = [event.event_id for event in events]
    if len(identifiers) != len(set(identifiers)):
        raise ValueError(f"duplicate runtime event_id in {path}")
    return events


class RuntimeEvidenceAugmenter:
    """Attach per-test dynamic evidence without exposing future patch data."""

    def apply(
        self,
        graph: RepositoryGraph,
        events: tuple[RuntimeEvent, ...],
    ) -> RepositoryGraph:
        nodes = {node.node_id: node for node in graph.nodes}
        edges = {edge.edge_id: edge for edge in graph.edges}
        evidence = {item.evidence_id: item for item in graph.evidence}
        test_nodes = {str(node.symbol): node for node in graph.nodes if node.kind == "test" and node.symbol}
        runtime_nodes = {str(node.symbol): node for node in graph.nodes if node.kind == "runtime_exception" and node.symbol}
        event_tests = {event.test_id for event in events}
        unknown_tests = sorted(event_tests - set(runtime_nodes))
        if unknown_tests:
            raise ValueError(f"runtime events reference unregistered failing tests: {unknown_tests}")

        # Replace the old combined traceback fan-out for tests with exact events.
        for edge_id, edge in tuple(edges.items()):
            target = nodes.get(edge.target)
            if edge.relation == "produces" and target is not None and target.kind == "runtime_exception" and target.symbol in event_tests:
                del edges[edge_id]

        for event in events:
            source = self._find_node(nodes.values(), event.source_file, event.source_symbol)
            if source is None:
                raise ValueError(f"runtime event source is absent from graph: {event.source_file}:{event.source_symbol}")
            ref = self._evidence_ref(event)
            evidence[ref.evidence_id] = ref
            runtime = runtime_nodes[event.test_id]
            test = test_nodes.get(event.test_id)

            if event.kind == "traceback_frame":
                self._add_edge(edges, source, runtime, "produces", ref)
            elif event.kind == "coverage":
                if test is None:
                    raise ValueError(f"failing test node is absent: {event.test_id}")
                self._add_edge(edges, test, source, "executes", ref)
            elif event.kind == "call":
                target = self._required_target(nodes, event)
                self._add_edge(edges, source, target, "runtime_calls", ref)
            elif event.kind in {"read", "write"}:
                target = self._artifact_target(nodes, source, event)
                relation = "reads" if event.kind == "read" else "writes"
                self._add_edge(edges, source, target, relation, ref)
            elif event.kind == "import":
                target = self._context_target(nodes, source, event, "module")
                self._add_edge(edges, source, target, "runtime_imports", ref)
            elif event.kind == "config_read":
                target = self._context_target(
                    nodes,
                    source,
                    event,
                    "configuration_key",
                )
                self._add_edge(edges, source, target, "configured_by", ref)
            elif event.kind == "dependency":
                target = self._context_target(nodes, source, event, "dependency")
                self._add_edge(edges, source, target, "depends_on", ref)
            else:
                nodes[runtime.node_id] = replace(
                    runtime,
                    evidence_refs=tuple(sorted({*runtime.evidence_refs, ref.evidence_id})),
                )

        return RepositoryGraph(
            graph.repository,
            graph.revision,
            tuple(sorted(nodes.values(), key=lambda item: item.node_id)),
            tuple(sorted(edges.values(), key=lambda item: item.edge_id)),
            tuple(sorted(evidence.values(), key=lambda item: item.evidence_id)),
            graph.obligations,
            graph.limitations,
        )

    @staticmethod
    def _find_node(
        nodes: Iterable[RepositoryNode],
        file_path: str,
        symbol: str | None,
    ) -> RepositoryNode | None:
        candidates = [
            node
            for node in nodes
            if node.file_path == file_path and (symbol is None or node.symbol == symbol or (node.symbol or "").rsplit(".", 1)[-1] == symbol)
        ]
        if symbol is not None:
            symbolic = next((node for node in candidates if node.symbol), None)
            if symbolic is not None:
                return symbolic
        return next(iter(candidates), None)

    def _required_target(
        self,
        nodes: dict[str, RepositoryNode],
        event: RuntimeEvent,
    ) -> RepositoryNode:
        if event.target_file is None:
            raise ValueError(f"{event.kind} event requires target_file")
        target = self._find_node(nodes.values(), event.target_file, event.target_symbol)
        if target is None:
            raise ValueError(f"runtime event target is absent from graph: {event.target_file}:{event.target_symbol}")
        return target

    def _artifact_target(
        self,
        nodes: dict[str, RepositoryNode],
        source: RepositoryNode,
        event: RuntimeEvent,
    ) -> RepositoryNode:
        if event.target_file is None:
            raise ValueError(f"{event.kind} event requires target_file")
        target = self._find_node(nodes.values(), event.target_file, event.target_symbol)
        if target is not None:
            return target
        node_id = "runtime_artifact:" + hashlib.sha256(event.target_file.encode()).hexdigest()[:16]
        target = RepositoryNode(
            node_id,
            "serialized_artifact",
            source.repository,
            event.target_file,
            event.target_symbol,
            {"observed_at_runtime": True},
        )
        nodes[node_id] = target
        return target

    def _context_target(
        self,
        nodes: dict[str, RepositoryNode],
        source: RepositoryNode,
        event: RuntimeEvent,
        kind: str,
    ) -> RepositoryNode:
        if event.target_file is None:
            raise ValueError(f"{event.kind} event requires target_file")
        target = self._find_node(
            nodes.values(),
            event.target_file,
            event.target_symbol,
        )
        if target is not None:
            return target
        node_id = f"runtime_{kind}:" + hashlib.sha256(f"{event.target_file}\0{event.target_symbol}".encode()).hexdigest()[:16]
        target = RepositoryNode(
            node_id,
            kind,
            source.repository,
            event.target_file,
            event.target_symbol,
            {"observed_at_runtime": True, "observed_value": event.detail},
        )
        nodes[node_id] = target
        return target

    @staticmethod
    def _evidence_ref(event: RuntimeEvent) -> EvidenceRef:
        evidence_id = (
            "evidence:runtime_event:"
            + hashlib.sha256(
                (
                    f"{event.event_id}\0{event.test_id}\0{event.kind}\0"
                    f"{event.source_file}\0{event.source_symbol}\0"
                    f"{event.target_file}\0{event.target_symbol}\0{event.detail}"
                ).encode()
            ).hexdigest()[:20]
        )
        return EvidenceRef(
            evidence_id,
            f"runtime_{event.kind}",
            event.source_file,
            event.detail or f"{event.kind} observed for {event.test_id}",
        )

    @staticmethod
    def _add_edge(
        edges: dict[str, RepositoryEdge],
        source: RepositoryNode,
        target: RepositoryNode,
        relation: str,
        evidence: EvidenceRef,
    ) -> None:
        edge_id = f"runtime-edge:{source.node_id}->{target.node_id}:{relation}:{evidence.evidence_id.rsplit(':', 1)[-1]}"
        edges[edge_id] = RepositoryEdge(
            edge_id,
            source.node_id,
            target.node_id,
            relation,
            (evidence.evidence_id,),
        )
