from __future__ import annotations

import copy
import hashlib
import json
from typing import Any

from .diagnostic_cut import DiagnosticCutSolver


def _canonical(value: dict[str, Any]) -> str:
    return json.dumps(value, ensure_ascii=True, sort_keys=True, separators=(",", ":"))


class FullH10GoldAuditor:
    """Graph-facing adapter for the frozen H10 cut and repair architecture.

    The adapter consumes only registered and observed routes. It has no access
    to mutation logs, oracle truth, case labels, or held-out fault names.
    """

    name = "full_h10"

    def __init__(self) -> None:
        self._solver = DiagnosticCutSolver(exact_node_limit=24)

    @staticmethod
    def _meaningful_changes(case: dict[str, Any]) -> tuple[list[str], list[str], bool]:
        expected = {node["id"]: node for node in case["registered_graph"]["nodes"]}
        observed = {node["id"]: node for node in case["observed_graph"]["nodes"]}
        direct: list[str] = []
        symptomatic: list[str] = []
        insufficient = False
        for node_id in sorted(set(expected) | set(observed)):
            before, after = expected.get(node_id), observed.get(node_id)
            token = f"node:{node_id}"
            if before is None or after is None:
                direct.append(token)
                continue
            changed_fields = {field for field in set(before) | set(after) if before.get(field) != after.get(field)}
            if not changed_fields:
                continue
            if any(after.get(field) is None for field in changed_fields):
                insufficient = True
            if changed_fields <= {"derived_status"}:
                symptomatic.append(token)
            else:
                direct.append(token)
        expected_edges = {(edge["source"], edge["target"]) for edge in case["registered_graph"]["edges"]}
        observed_edges = {(edge["source"], edge["target"]) for edge in case["observed_graph"]["edges"]}
        direct.extend(f"edge:{source}->{target}" for source, target in sorted(expected_edges ^ observed_edges))
        return sorted(set(direct)), sorted(set(symptomatic)), insufficient

    @staticmethod
    def _paths(case: dict[str, Any], direct: list[str]) -> tuple[frozenset[str], ...]:
        registered = case["registered_graph"]
        result: list[frozenset[str]] = []
        for path in registered["audit_paths"]:
            tokens: set[str] = set()
            for node_id in path:
                token = f"node:{node_id}"
                if token in direct:
                    tokens.add(token)
            for source, target in zip(path, path[1:]):
                token = f"edge:{source}->{target}"
                if token in direct:
                    tokens.add(token)
            if tokens:
                result.append(frozenset(tokens))
        for token in direct:
            if not any(token in path for path in result):
                result.append(frozenset((token,)))
        return tuple(result)

    @staticmethod
    def _repairs(case: dict[str, Any], sources: list[str]) -> list[str]:
        registered = case["registered_graph"]
        observed = case["observed_graph"]
        expected_nodes = {node["id"]: node for node in registered["nodes"]}
        actual_nodes = {node["id"]: node for node in observed["nodes"]}
        repairs: list[str] = []
        for token in sources:
            if token.startswith("edge:"):
                source, target = token.split(":", 1)[1].split("->", 1)
                expected_edge = any(edge["source"] == source and edge["target"] == target for edge in registered["edges"])
                operation = "add_edge" if expected_edge else "remove_edge"
                repairs.append(_canonical({"operation": operation, "source": source, "target": target}))
                continue
            node_id = token.split(":", 1)[1]
            before, after = expected_nodes.get(node_id), actual_nodes.get(node_id)
            if before is None:
                repairs.append(_canonical({"operation": "remove_node", "node_id": node_id}))
            elif after is None:
                repairs.append(_canonical({"operation": "restore_node", "node": before}))
                repairs.extend(
                    _canonical({"operation": "add_edge", "source": edge["source"], "target": edge["target"]})
                    for edge in registered["edges"]
                    if node_id in (edge["source"], edge["target"])
                )
            else:
                for field in sorted(set(before) | set(after)):
                    if field != "derived_status" and before.get(field) != after.get(field):
                        repairs.append(
                            _canonical({"operation": "restore_attribute", "node_id": node_id, "field": field, "value": before.get(field)})
                        )
        return sorted(set(repairs))

    @staticmethod
    def _recertify(case: dict[str, Any], repairs: list[str]) -> bool:
        graph = copy.deepcopy(case["observed_graph"])
        nodes = {node["id"]: node for node in graph["nodes"]}
        for encoded in repairs:
            action = json.loads(encoded)
            operation = action["operation"]
            if operation == "restore_attribute" and action["node_id"] in nodes:
                nodes[action["node_id"]][action["field"]] = action["value"]
            elif operation == "restore_node":
                node = copy.deepcopy(action["node"])
                graph["nodes"] = [item for item in graph["nodes"] if item["id"] != node["id"]] + [node]
                nodes[node["id"]] = node
            elif operation == "remove_node":
                graph["nodes"] = [item for item in graph["nodes"] if item["id"] != action["node_id"]]
            elif operation == "add_edge":
                if not any(edge["source"] == action["source"] and edge["target"] == action["target"] for edge in graph["edges"]):
                    graph["edges"].append({"source": action["source"], "target": action["target"]})
            elif operation == "remove_edge":
                graph["edges"] = [
                    edge for edge in graph["edges"]
                    if not (edge["source"] == action["source"] and edge["target"] == action["target"])
                ]
        expected = copy.deepcopy(case["registered_graph"])
        for node in graph["nodes"]:
            if node.get("derived_status") == "upstream_contract_unresolved":
                node["derived_status"] = "consistent"
        def normalize(value: dict[str, Any]) -> str:
            return _canonical(
                {
                    **value,
                    "nodes": sorted(value["nodes"], key=lambda item: item["id"]),
                    "edges": sorted(value["edges"], key=lambda item: (item["source"], item["target"])),
                }
            )

        return normalize(graph) == normalize(expected)

    def diagnose(self, case: dict[str, Any]) -> dict[str, Any]:
        direct, symptomatic, insufficient = self._meaningful_changes(case)
        if not direct and not symptomatic:
            status = "valid"
            cut_nodes: list[str] = []
            repairs: list[str] = []
            recertified = True
        else:
            status = "insufficient_evidence" if insufficient else "invalid"
            paths = self._paths(case, direct)
            costs = {key: float(value) for key, value in case["repair_costs"].items()}
            result = self._solver.solve(paths, costs) if paths else None
            cut_nodes = list(result.cut_nodes) if result else []
            repairs = self._repairs(case, direct)
            recertified = self._recertify(case, repairs) if repairs else False
        payload = {
            "case_id": case["case_id"],
            "route_status": status,
            "source_elements": direct,
            "cut_nodes": cut_nodes,
            "repair_actions": repairs,
            "recertified": recertified,
            "abstained": insufficient,
        }
        trace = _canonical(payload).encode("ascii")
        payload["trace"] = hashlib.sha256(trace).hexdigest()
        payload["method"] = self.name
        return payload
