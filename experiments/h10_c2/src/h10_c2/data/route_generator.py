from __future__ import annotations

from typing import Any


NODE_TYPES = (
    "data",
    "preprocessing",
    "model",
    "calibration",
    "explainer",
    "representation",
    "reduction",
    "serialization",
    "provenance",
    "runtime",
)


def build_clean_route(pipeline_id: str, width: int, case_index: int) -> dict[str, Any]:
    nodes = []
    for index in range(width):
        node_type = NODE_TYPES[index % len(NODE_TYPES)]
        node_id = f"{node_type}:{index}"
        registered = {
            "version": f"v{1 + index % 4}",
            "schema": f"schema-{pipeline_id}-{index}",
            "sha256": f"sha-{pipeline_id}-{case_index}-{index}",
            "available": True,
        }
        nodes.append(
            {
                "node_id": node_id,
                "node_type": node_type,
                "component_version": registered["version"],
                "registered_attributes": registered,
                "observed_attributes": dict(registered),
                "mandatory": True,
                "repairable": index != width - 1,
                "evidence_refs": [f"evidence:{pipeline_id}:{case_index}:{index}"],
            }
        )
    edges = []
    for index in range(width - 1):
        edges.append(
            {
                "edge_id": f"edge:{index}->{index + 1}",
                "source": nodes[index]["node_id"],
                "target": nodes[index + 1]["node_id"],
                "relation": "transforms",
                "mandatory": True,
                "registered_contract": {"compatible": True},
                "observed_contract": {"compatible": True},
                "repairable": True,
            }
        )
    if width >= 6:
        edges.append(
            {
                "edge_id": "edge:branch",
                "source": nodes[1]["node_id"],
                "target": nodes[-2]["node_id"],
                "relation": "derived_from",
                "mandatory": True,
                "registered_contract": {"compatible": True},
                "observed_contract": {"compatible": True},
                "repairable": True,
            }
        )
    return {
        "route_id": f"route:{pipeline_id}:{case_index}",
        "nodes": nodes,
        "edges": edges,
        "metadata": {"pipeline": pipeline_id},
    }

