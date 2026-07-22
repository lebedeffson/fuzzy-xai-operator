from __future__ import annotations

import copy
from dataclasses import dataclass
from typing import Any

from .graph_diff import diff_graphs


@dataclass(frozen=True)
class MutationTransaction:
    transaction_id: str
    operation: str
    parameters: dict[str, Any]
    changed_nodes: tuple[str, ...]
    changed_edges: tuple[str, ...]
    inverse_operation: dict[str, Any]

    def as_dict(self) -> dict[str, Any]:
        return {
            "transaction_id": self.transaction_id,
            "operation": self.operation,
            "parameters": self.parameters,
            "changed_nodes": list(self.changed_nodes),
            "changed_edges": list(self.changed_edges),
            "inverse_operation": self.inverse_operation,
        }


def _nodes(graph: dict[str, Any]) -> dict[str, dict[str, Any]]:
    return {str(node["id"]): node for node in graph["nodes"]}


def _remove_edge(graph: dict[str, Any], source: str, target: str) -> None:
    graph["edges"] = [
        edge for edge in graph["edges"]
        if not (edge["source"] == source and edge["target"] == target)
    ]


def _add_edge(graph: dict[str, Any], source: str, target: str) -> None:
    if not any(edge["source"] == source and edge["target"] == target for edge in graph["edges"]):
        graph["edges"].append({"source": source, "target": target})


def apply_transaction(
    graph: dict[str, Any],
    *,
    transaction_id: str,
    operation: str,
    parameters: dict[str, Any],
) -> tuple[dict[str, Any], MutationTransaction]:
    """Apply one low-level mutation and derive its direct truth from graph delta."""
    before = copy.deepcopy(graph)
    after = copy.deepcopy(graph)
    nodes = _nodes(after)
    inverse: dict[str, Any]

    if operation == "remove_node":
        node_id = str(parameters["node_id"])
        original = copy.deepcopy(nodes[node_id])
        incident = [copy.deepcopy(edge) for edge in after["edges"] if node_id in (edge["source"], edge["target"])]
        after["nodes"] = [node for node in after["nodes"] if node["id"] != node_id]
        after["edges"] = [edge for edge in after["edges"] if node_id not in (edge["source"], edge["target"])]
        inverse = {"operation": "restore_node", "node": original, "edges": incident}
    elif operation in {
        "corrupt_checksum",
        "replace_model_binding",
        "replace_node_version",
        "mix_run_artifact",
        "replace_preprocessing",
        "replace_reference_population",
        "replace_explainer_version",
        "replace_feature_schema",
    }:
        node_id = str(parameters["node_id"])
        field = str(parameters["field"])
        old_value = copy.deepcopy(nodes[node_id].get(field))
        nodes[node_id][field] = copy.deepcopy(parameters["value"])
        inverse = {"operation": "restore_attribute", "node_id": node_id, "field": field, "value": old_value}
    elif operation == "remove_edge":
        source, target = str(parameters["source"]), str(parameters["target"])
        _remove_edge(after, source, target)
        inverse = {"operation": "add_edge", "source": source, "target": target}
    elif operation == "add_false_dependency":
        source, target = str(parameters["source"]), str(parameters["target"])
        _add_edge(after, source, target)
        inverse = {"operation": "remove_edge", "source": source, "target": target}
    elif operation == "reorder_components":
        first, second, third = (str(parameters[key]) for key in ("first", "second", "third"))
        _remove_edge(after, first, second)
        _remove_edge(after, second, third)
        _add_edge(after, first, third)
        _add_edge(after, third, second)
        inverse = {
            "operation": "restore_edges",
            "remove": [[first, third], [third, second]],
            "add": [[first, second], [second, third]],
        }
    else:
        raise ValueError(f"unsupported low-level operation: {operation}")

    delta = diff_graphs(before, after)
    changed_edges = tuple(dict.fromkeys(delta.added_edges + delta.removed_edges))
    transaction = MutationTransaction(
        transaction_id=transaction_id,
        operation=operation,
        parameters=copy.deepcopy(parameters),
        changed_nodes=tuple(dict.fromkeys(delta.added_nodes + delta.removed_nodes + delta.changed_nodes)),
        changed_edges=changed_edges,
        inverse_operation=inverse,
    )
    return after, transaction
