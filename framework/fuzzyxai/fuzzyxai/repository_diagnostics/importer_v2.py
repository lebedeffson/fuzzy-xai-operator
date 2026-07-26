from __future__ import annotations

import ast
import hashlib
import warnings
from dataclasses import replace

from .graph import EvidenceRef, RepositoryEdge, RepositoryGraph, RepositoryNode
from .importer import RepositoryIncident, RepositoryStructureImporter
from .runtime_events import RuntimeEvent, RuntimeEvidenceAugmenter

PARSEABLE_SOURCE = "PARSEABLE_SOURCE"
INTENTIONAL_SYNTAX_FIXTURE = "INTENTIONAL_SYNTAX_FIXTURE"
UNPARSEABLE_RUNTIME_TARGET = "UNPARSEABLE_RUNTIME_TARGET"
UNPARSEABLE_NON_RUNTIME_SOURCE = "UNPARSEABLE_NON_RUNTIME_SOURCE"


def _intentional_fixture(path: str) -> bool:
    normalized = path.lower().replace("\\", "/")
    name = normalized.rsplit("/", 1)[-1]
    return (
        any(
            marker in normalized
            for marker in (
                "/fixtures/",
                "/testdata/",
                "/test_data/",
                "/tests/data/",
                "/tests/functional/",
            )
        )
        or name in {"bad.py", "invalid.py", "syntax_error.py"}
        or "syntax_fixture" in name
    )


class EvidenceGroundedRepositoryImporter:
    """Add fail-closed parse states without changing the frozen H10-C5b importer."""

    def __init__(self, *, max_python_files: int = 4000) -> None:
        self.base = RepositoryStructureImporter(max_python_files=max_python_files)

    def build(
        self,
        incident: RepositoryIncident,
        *,
        runtime_events: tuple[RuntimeEvent, ...] = (),
    ) -> RepositoryGraph:
        graph = self.base.build(incident)
        nodes = {node.node_id: node for node in graph.nodes}
        edges = {edge.edge_id: edge for edge in graph.edges}
        evidence = {item.evidence_id: item for item in graph.evidence}
        runtime_text = (f"{incident.traceback}\n{incident.stdout}\n{incident.stderr}\n{' '.join(incident.failing_tests)}").replace("\\", "/")
        limitations = [value for value in graph.limitations if not value.startswith("syntax_error:")]

        for node in tuple(nodes.values()):
            if node.kind != "file" or not (node.file_path or "").endswith(".py"):
                continue
            path = incident.repository_root / str(node.file_path)
            source = path.read_text(encoding="utf-8", errors="replace")
            try:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore", SyntaxWarning)
                    ast.parse(source, filename=str(node.file_path))
                status = PARSEABLE_SOURCE
                detail = "source parsed successfully"
            except SyntaxError as error:
                if _intentional_fixture(str(node.file_path)):
                    status = INTENTIONAL_SYNTAX_FIXTURE
                elif str(node.file_path) in runtime_text:
                    status = UNPARSEABLE_RUNTIME_TARGET
                    limitations.append(f"unparseable_runtime_target:{node.file_path}:{error.lineno}")
                else:
                    status = UNPARSEABLE_NON_RUNTIME_SOURCE
                detail = f"{status}: line {error.lineno}"
                ref = "evidence:parse_state:" + hashlib.sha256(f"{node.file_path}\0{status}\0{error.lineno}".encode()).hexdigest()[:16]
                evidence[ref] = EvidenceRef(
                    ref,
                    "parse_state",
                    str(node.file_path),
                    detail,
                )
                state_id = f"source_parse_state:{node.file_path}"
                nodes[state_id] = RepositoryNode(
                    state_id,
                    "source_parse_state",
                    node.repository,
                    node.file_path,
                    status,
                    {
                        "parse_status": status,
                        "lineno": error.lineno,
                        "message": error.msg,
                    },
                    (ref,),
                )
                edge_id = f"edge:{node.node_id}->{state_id}:has_parse_state"
                edges[edge_id] = RepositoryEdge(
                    edge_id,
                    node.node_id,
                    state_id,
                    "has_parse_state",
                    (ref,),
                )
            nodes[node.node_id] = replace(
                node,
                attributes={**node.attributes, "parse_status": status},
            )

        result = RepositoryGraph(
            graph.repository,
            graph.revision,
            tuple(sorted(nodes.values(), key=lambda item: item.node_id)),
            tuple(sorted(edges.values(), key=lambda item: item.edge_id)),
            tuple(sorted(evidence.values(), key=lambda item: item.evidence_id)),
            graph.obligations,
            tuple(sorted(set(limitations))),
        )
        if runtime_events:
            result = RuntimeEvidenceAugmenter().apply(result, runtime_events)
        return result
