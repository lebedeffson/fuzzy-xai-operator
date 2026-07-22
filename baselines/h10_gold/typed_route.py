from __future__ import annotations

import json
from typing import Any


def _canonical(value: dict[str, Any]) -> str:
    return json.dumps(value, ensure_ascii=True, sort_keys=True, separators=(",", ":"))


class TypedRouteWithoutReasoning:
    """Contract validator without cut optimization or minimal repair reasoning."""

    name = "typed_route_without_cut"

    def diagnose(self, case: dict[str, Any]) -> dict[str, Any]:
        registered = case["registered_graph"]
        observed = case["observed_graph"]
        expected_nodes = {item["id"]: item for item in registered["nodes"]}
        actual_nodes = {item["id"]: item for item in observed["nodes"]}
        violations: list[str] = []
        repairs: list[str] = []
        insufficient = False
        for path in registered["audit_paths"]:
            for node_id in path:
                expected = expected_nodes.get(node_id)
                actual = actual_nodes.get(node_id)
                if expected is None:
                    continue
                if actual is None:
                    violations.append(f"node:{node_id}")
                    repairs.append(_canonical({"operation": "restore_node", "node": expected}))
                    continue
                if actual.get("version") is None or actual.get("checksum") is None:
                    insufficient = True
                if expected != actual:
                    actionable_fields = [
                        field for field in sorted(set(expected) | set(actual))
                        if field != "derived_status" and expected.get(field) != actual.get(field)
                    ]
                    if actionable_fields:
                        violations.append(f"node:{node_id}")
                    for field in sorted(set(expected) | set(actual)):
                        if field != "derived_status" and expected.get(field) != actual.get(field):
                            repairs.append(
                                _canonical({"operation": "restore_attribute", "node_id": node_id, "field": field, "value": expected.get(field)})
                            )
            for source, target in zip(path, path[1:]):
                expected_edge = (source, target)
                actual_edges = {(edge["source"], edge["target"]) for edge in observed["edges"]}
                if expected_edge not in actual_edges:
                    violations.append(f"edge:{source}->{target}")
                    repairs.append(_canonical({"operation": "add_edge", "source": source, "target": target}))
        expected_edges = {(edge["source"], edge["target"]) for edge in registered["edges"]}
        actual_edges = {(edge["source"], edge["target"]) for edge in observed["edges"]}
        for source, target in sorted(actual_edges - expected_edges):
            violations.append(f"edge:{source}->{target}")
            repairs.append(_canonical({"operation": "remove_edge", "source": source, "target": target}))
        status = "valid" if not violations else ("insufficient_evidence" if insufficient else "invalid")
        return {
            "method": self.name,
            "route_status": status,
            "source_elements": sorted(set(violations)),
            "cut_nodes": sorted(set(violations)),
            "repair_actions": sorted(set(repairs)),
            "abstained": insufficient,
            "trace": "",
        }
