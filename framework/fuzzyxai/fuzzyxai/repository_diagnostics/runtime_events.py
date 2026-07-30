from __future__ import annotations

import hashlib
import json
from collections import defaultdict
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, replace
from pathlib import Path

from .graph import EvidenceRef, RepositoryEdge, RepositoryGraph, RepositoryNode
from .importer import FORBIDDEN_GOLD_FIELDS

EVENT_KINDS = frozenset(
    {
        "assertion",
        "assertion_operand",
        "argument_value",
        "call",
        "config_read",
        "coverage",
        "dependency",
        "exception",
        "import",
        "read",
        "last_writer",
        "return_value",
        "traceback_frame",
        "value_flow",
        "write",
    }
)


def _is_test_path(path: str) -> bool:
    normalized = path.replace("\\", "/").lower()
    return (
        normalized.startswith(("test/", "tests/"))
        or "/tests/" in normalized
        or "/test_" in normalized
        or normalized.endswith("_test.py")
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
    sequence_id: int = -1
    timestamp_ns: int = 0
    thread_id: int = 0
    call_depth: int = 0
    occurrence_count: int = 1
    first_sequence_id: int = -1
    last_sequence_id: int = -1

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
            sequence_id=int(value.get("sequence_id", -1)),
            timestamp_ns=int(value.get("timestamp_ns", 0)),
            thread_id=int(value.get("thread_id", 0)),
            call_depth=max(0, int(value.get("call_depth", 0))),
            occurrence_count=max(1, int(value.get("occurrence_count", 1))),
            first_sequence_id=int(
                value.get(
                    "first_sequence_id",
                    value.get("sequence_id", -1),
                )
            ),
            last_sequence_id=int(
                value.get(
                    "last_sequence_id",
                    value.get("sequence_id", -1),
                )
            ),
        )
        if not event.event_id or not event.test_id or not event.source_file:
            raise ValueError("runtime event requires event_id, test_id and source_file")
        return event


def _optional_text(value: object) -> str | None:
    text = str(value or "").strip()
    return text.replace("\\", "/") if text else None


def load_runtime_events(path: Path) -> tuple[RuntimeEvent, ...]:
    events = tuple(
        _with_physical_sequence(
            RuntimeEvent.from_mapping(json.loads(line)),
            index,
        )
        for index, line in enumerate(
            line
            for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        )
    )
    identifiers = [event.event_id for event in events]
    if len(identifiers) != len(set(identifiers)):
        raise ValueError(f"duplicate runtime event_id in {path}")
    return events


def _with_physical_sequence(event: RuntimeEvent, index: int) -> RuntimeEvent:
    if event.sequence_id >= 0:
        return event
    return replace(
        event,
        sequence_id=index,
        first_sequence_id=(
            index if event.first_sequence_id < 0 else event.first_sequence_id
        ),
        last_sequence_id=(
            index if event.last_sequence_id < 0 else event.last_sequence_id
        ),
    )


def normalize_runtime_event_rows(
    rows: Sequence[dict[str, object]],
) -> list[dict[str, object]]:
    """Preserve physical order while making per-test sequences monotonic."""
    next_sequence: dict[str, int] = defaultdict(int)
    seen_ids: set[str] = set()
    normalized: list[dict[str, object]] = []
    for physical_index, source in enumerate(rows):
        row = dict(source)
        test_id = str(row.get("test_id", ""))
        requested = int(row.get("sequence_id", -1))
        sequence = max(next_sequence[test_id], requested)
        first = int(row.get("first_sequence_id", requested))
        last = int(row.get("last_sequence_id", requested))
        width = max(0, last - first)
        row["sequence_id"] = sequence
        row["first_sequence_id"] = sequence
        row["last_sequence_id"] = sequence + width
        row["timestamp_ns"] = int(row.get("timestamp_ns", 0))
        row["thread_id"] = int(row.get("thread_id", 0))
        row["call_depth"] = max(0, int(row.get("call_depth", 0)))
        row["occurrence_count"] = max(
            1,
            int(row.get("occurrence_count", 1)),
        )
        next_sequence[test_id] = row["last_sequence_id"] + 1
        event_id = str(row.get("event_id", ""))
        if event_id in seen_ids:
            row["event_id"] = event_id + "-" + hashlib.sha256(
                f"{test_id}\0{sequence}\0{physical_index}".encode()
            ).hexdigest()[:8]
        seen_ids.add(str(row["event_id"]))
        normalized.append(row)
    return normalized


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
            ref = self._evidence_ref(event)
            evidence[ref.evidence_id] = ref
            source = self._runtime_symbol(
                nodes,
                edges,
                event.source_file,
                event.source_symbol,
                ref,
                role="source",
            )
            runtime = runtime_nodes[event.test_id]
            test = self._test_for_runtime(nodes, edges, runtime)

            if event.kind == "traceback_frame":
                self._add_edge(edges, source, runtime, "produces", ref)
            elif event.kind == "coverage":
                if test is None:
                    raise ValueError(f"failing test node is absent: {event.test_id}")
                self._add_edge(edges, test, source, "executes", ref)
            elif event.kind == "call":
                target = self._required_target(nodes, edges, event, ref)
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
            elif event.kind in {
                "argument_value",
                "return_value",
                "assertion_operand",
                "exception",
            }:
                target = self._value_target(nodes, source, event, ref)
                relation = {
                    "argument_value": "consumes",
                    "return_value": "produces",
                    "assertion_operand": "asserts_on",
                    "exception": "raises",
                }[event.kind]
                self._add_edge(edges, source, target, relation, ref)
            elif event.kind in {"last_writer", "value_flow"}:
                target = self._required_target(nodes, edges, event, ref)
                relation = (
                    "last_writes"
                    if event.kind == "last_writer"
                    else "value_flows_to"
                )
                self._add_edge(edges, source, target, relation, ref)
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
    def _test_for_runtime(
        nodes: dict[str, RepositoryNode],
        edges: dict[str, RepositoryEdge],
        runtime: RepositoryNode,
    ) -> RepositoryNode | None:
        matches = tuple(
            nodes[edge.source]
            for edge in edges.values()
            if edge.relation == "fails_in"
            and edge.target == runtime.node_id
            and edge.source in nodes
            and nodes[edge.source].kind == "test"
        )
        return matches[0] if len(matches) == 1 else None

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

    def _runtime_symbol(
        self,
        nodes: dict[str, RepositoryNode],
        edges: dict[str, RepositoryEdge],
        file_path: str,
        symbol: str | None,
        evidence: EvidenceRef,
        *,
        role: str,
    ) -> RepositoryNode:
        node = self._find_node(nodes.values(), file_path, symbol)
        if node is not None:
            return node
        file_node = next(
            (
                candidate
                for candidate in nodes.values()
                if candidate.kind == "file"
                and candidate.file_path == file_path
            ),
            None,
        )
        if file_node is None:
            if _is_test_path(file_path):
                node_id = "runtime_test_support:" + hashlib.sha256(
                    f"{file_path}\0{symbol}".encode()
                ).hexdigest()[:16]
                node = RepositoryNode(
                    node_id,
                    "runtime_test_support",
                    next(iter(nodes.values())).repository,
                    file_path,
                    symbol,
                    {"observed_at_runtime": True, "source_unavailable": True},
                    (evidence.evidence_id,),
                )
                nodes[node_id] = node
                return node
            raise ValueError(
                f"runtime event {role} is absent from graph: "
                f"{file_path}:{symbol}"
            )
        if symbol is None:
            return file_node
        node_id = "runtime_symbol:" + hashlib.sha256(
            f"{file_path}\0{symbol}".encode()
        ).hexdigest()[:16]
        node = RepositoryNode(
            node_id,
            "runtime_symbol",
            file_node.repository,
            file_path,
            symbol,
            {"observed_at_runtime": True},
            (evidence.evidence_id,),
        )
        nodes[node_id] = node
        self._add_edge(edges, file_node, node, "contains", evidence)
        return node

    def _required_target(
        self,
        nodes: dict[str, RepositoryNode],
        edges: dict[str, RepositoryEdge],
        event: RuntimeEvent,
        evidence: EvidenceRef,
    ) -> RepositoryNode:
        if event.target_file is None:
            raise ValueError(f"{event.kind} event requires target_file")
        return self._runtime_symbol(
            nodes,
            edges,
            event.target_file,
            event.target_symbol,
            evidence,
            role="target",
        )

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
    def _value_target(
        nodes: dict[str, RepositoryNode],
        source: RepositoryNode,
        event: RuntimeEvent,
        evidence: EvidenceRef,
    ) -> RepositoryNode:
        try:
            payload = json.loads(event.detail)
        except json.JSONDecodeError:
            payload = {"preview": event.detail}
        identity = str(
            payload.get("object_id")
            or payload.get("digest")
            or event.event_id
        )
        node_id = "runtime_value:" + hashlib.sha256(
            f"{event.test_id}\0{identity}".encode()
        ).hexdigest()[:20]
        existing = nodes.get(node_id)
        if existing is not None:
            nodes[node_id] = replace(
                existing,
                evidence_refs=tuple(
                    sorted(
                        {
                            *existing.evidence_refs,
                            evidence.evidence_id,
                        }
                    )
                ),
            )
            return nodes[node_id]
        safe_attributes = {
            key: value
            for key, value in payload.items()
            if key
            in {
                "digest",
                "fields",
                "length",
                "name",
                "object_id",
                "preview",
                "shape",
                "type",
                "writer_sequence_id",
            }
        }
        target = RepositoryNode(
            node_id,
            "runtime_value",
            source.repository,
            event.source_file,
            identity,
            {
                **safe_attributes,
                "observed_at_runtime": True,
                "test_id": event.test_id,
            },
            (evidence.evidence_id,),
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
                    f"{event.target_file}\0{event.target_symbol}\0{event.detail}\0"
                    f"{event.sequence_id}\0{event.occurrence_count}"
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
