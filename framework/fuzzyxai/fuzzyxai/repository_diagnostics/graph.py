from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from hashlib import sha256


@dataclass(frozen=True)
class EvidenceRef:
    evidence_id: str
    kind: str
    source: str
    detail: str


@dataclass(frozen=True)
class RepositoryNode:
    node_id: str
    kind: str
    repository: str
    file_path: str | None = None
    symbol: str | None = None
    attributes: dict[str, object] = field(default_factory=dict)
    evidence_refs: tuple[str, ...] = ()


@dataclass(frozen=True)
class RepositoryEdge:
    edge_id: str
    source: str
    target: str
    relation: str
    evidence_refs: tuple[str, ...] = ()


@dataclass(frozen=True)
class RepositoryGraph:
    repository: str
    revision: str
    nodes: tuple[RepositoryNode, ...]
    edges: tuple[RepositoryEdge, ...]
    evidence: tuple[EvidenceRef, ...]
    obligations: tuple[str, ...]
    limitations: tuple[str, ...] = ()

    def node(self, node_id: str) -> RepositoryNode | None:
        return next((node for node in self.nodes if node.node_id == node_id), None)

    def predecessors(self, node_id: str) -> tuple[RepositoryNode, ...]:
        ids = {edge.source for edge in self.edges if edge.target == node_id}
        return tuple(node for node in self.nodes if node.node_id in ids)

    def successors(self, node_id: str) -> tuple[RepositoryNode, ...]:
        ids = {edge.target for edge in self.edges if edge.source == node_id}
        return tuple(node for node in self.nodes if node.node_id in ids)

    @property
    def trace_sha256(self) -> str:
        payload = {
            "repository": self.repository,
            "revision": self.revision,
            "nodes": [asdict(node) for node in self.nodes],
            "edges": [asdict(edge) for edge in self.edges],
            "evidence": [asdict(item) for item in self.evidence],
            "obligations": self.obligations,
            "limitations": self.limitations,
        }
        canonical = json.dumps(
            payload,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
        ).encode()
        return sha256(canonical).hexdigest()
