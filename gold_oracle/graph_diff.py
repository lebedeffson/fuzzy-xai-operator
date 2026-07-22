from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable


def node_token(node_id: str) -> str:
    return f"node:{node_id}"


def edge_token(source: str, target: str) -> str:
    return f"edge:{source}->{target}"


def _node_map(graph: dict[str, Any]) -> dict[str, dict[str, Any]]:
    return {str(node["id"]): node for node in graph.get("nodes", [])}


def _edge_set(graph: dict[str, Any]) -> set[tuple[str, str]]:
    return {(str(edge["source"]), str(edge["target"])) for edge in graph.get("edges", [])}


@dataclass(frozen=True)
class GraphDifference:
    added_nodes: tuple[str, ...]
    removed_nodes: tuple[str, ...]
    changed_nodes: tuple[str, ...]
    added_edges: tuple[str, ...]
    removed_edges: tuple[str, ...]

    @property
    def changed_elements(self) -> tuple[str, ...]:
        return tuple(
            dict.fromkeys(
                self.added_nodes
                + self.removed_nodes
                + self.changed_nodes
                + self.added_edges
                + self.removed_edges
            )
        )


def diff_graphs(clean: dict[str, Any], mutated: dict[str, Any]) -> GraphDifference:
    clean_nodes = _node_map(clean)
    mutated_nodes = _node_map(mutated)
    added_node_ids = sorted(set(mutated_nodes) - set(clean_nodes))
    removed_node_ids = sorted(set(clean_nodes) - set(mutated_nodes))
    changed_node_ids = sorted(
        node_id
        for node_id in set(clean_nodes) & set(mutated_nodes)
        if clean_nodes[node_id] != mutated_nodes[node_id]
    )
    clean_edges = _edge_set(clean)
    mutated_edges = _edge_set(mutated)
    return GraphDifference(
        added_nodes=tuple(node_token(item) for item in added_node_ids),
        removed_nodes=tuple(node_token(item) for item in removed_node_ids),
        changed_nodes=tuple(node_token(item) for item in changed_node_ids),
        added_edges=tuple(edge_token(*item) for item in sorted(mutated_edges - clean_edges)),
        removed_edges=tuple(edge_token(*item) for item in sorted(clean_edges - mutated_edges)),
    )


def path_tokens(path: Iterable[str]) -> tuple[str, ...]:
    nodes = tuple(str(item) for item in path)
    tokens: list[str] = []
    for index, node_id in enumerate(nodes):
        tokens.append(node_token(node_id))
        if index + 1 < len(nodes):
            tokens.append(edge_token(node_id, nodes[index + 1]))
    return tuple(tokens)


def derive_broken_paths(clean: dict[str, Any], mutated: dict[str, Any]) -> tuple[tuple[str, ...], ...]:
    """Derive affected registered paths from graph state, not from fault names."""
    difference = diff_graphs(clean, mutated)
    clean_nodes = _node_map(clean)
    mutated_nodes = _node_map(mutated)
    meaningful_nodes = set(difference.added_nodes + difference.removed_nodes)
    for token in difference.changed_nodes:
        node_id = token.split(":", 1)[1]
        before = {key: value for key, value in clean_nodes[node_id].items() if key != "derived_status"}
        after = {key: value for key, value in mutated_nodes[node_id].items() if key != "derived_status"}
        if before != after:
            meaningful_nodes.add(token)
    changed = meaningful_nodes | set(difference.added_edges) | set(difference.removed_edges)
    paths = []
    for path in clean.get("audit_paths", []):
        tokens = path_tokens(path)
        affected = tuple(token for token in tokens if token in changed)
        if affected:
            paths.append(affected)
    registered_tokens = {token for path in paths for token in path}
    for token in difference.added_edges:
        if token not in registered_tokens:
            _, endpoints = token.split(":", 1)
            source, target = endpoints.split("->", 1)
            paths.append((node_token(source), token, node_token(target)))
    for token in changed:
        if not any(token in path for path in paths):
            paths.append((token,))
    return tuple(dict.fromkeys(paths))
