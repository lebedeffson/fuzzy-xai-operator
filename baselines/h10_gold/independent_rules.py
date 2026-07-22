from __future__ import annotations

import json
from typing import Any


def _canonical(action: dict[str, Any]) -> str:
    return json.dumps(action, ensure_ascii=True, sort_keys=True, separators=(",", ":"))


class IndependentIfElse:
    """Standalone rule engine intentionally unaware of H10 graph classes."""

    name = "independent_if_else"

    def diagnose(self, case: dict[str, Any]) -> dict[str, Any]:
        expected = {node["id"]: node for node in case["registered_graph"]["nodes"]}
        observed = {node["id"]: node for node in case["observed_graph"]["nodes"]}
        sources: list[str] = []
        repairs: list[str] = []
        insufficient = False
        for node_id in sorted(set(expected) | set(observed)):
            before, after = expected.get(node_id), observed.get(node_id)
            if before is None:
                sources.append(f"node:{node_id}")
                repairs.append(_canonical({"operation": "remove_node", "node_id": node_id}))
                continue
            if after is None:
                sources.append(f"node:{node_id}")
                repairs.append(_canonical({"operation": "restore_node", "node": before}))
                for edge in case["registered_graph"]["edges"]:
                    if node_id in (edge["source"], edge["target"]):
                        repairs.append(_canonical({"operation": "add_edge", "source": edge["source"], "target": edge["target"]}))
                continue
            for field in sorted(set(before) | set(after)):
                if before.get(field) == after.get(field):
                    continue
                # A strong rule baseline knows that this field is a propagated
                # symptom and must not propose repairing it directly.
                if field == "derived_status":
                    continue
                sources.append(f"node:{node_id}")
                if after.get(field) is None:
                    insufficient = True
                repairs.append(
                    _canonical({"operation": "restore_attribute", "node_id": node_id, "field": field, "value": before.get(field)})
                )
        expected_edges = {(edge["source"], edge["target"]) for edge in case["registered_graph"]["edges"]}
        observed_edges = {(edge["source"], edge["target"]) for edge in case["observed_graph"]["edges"]}
        for source, target in sorted(expected_edges - observed_edges):
            sources.append(f"edge:{source}->{target}")
            repairs.append(_canonical({"operation": "add_edge", "source": source, "target": target}))
        for source, target in sorted(observed_edges - expected_edges):
            sources.append(f"edge:{source}->{target}")
            repairs.append(_canonical({"operation": "remove_edge", "source": source, "target": target}))
        status = "valid" if not sources else ("insufficient_evidence" if insufficient else "invalid")
        return {
            "method": self.name,
            "route_status": status,
            "source_elements": sorted(set(sources)),
            "cut_nodes": sorted(set(sources)),
            "repair_actions": sorted(set(repairs)),
            "abstained": insufficient,
            "trace": "",
        }
