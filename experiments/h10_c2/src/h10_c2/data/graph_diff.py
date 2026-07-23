from __future__ import annotations


def changed_subjects(clean: dict, observed: dict) -> tuple[str, ...]:
    clean_nodes = {node["node_id"]: node for node in clean["nodes"]}
    observed_nodes = {node["node_id"]: node for node in observed["nodes"]}
    return tuple(sorted(node_id for node_id in clean_nodes.keys() | observed_nodes.keys() if clean_nodes.get(node_id) != observed_nodes.get(node_id)))

