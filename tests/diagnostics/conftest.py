from __future__ import annotations

import pytest

from fuzzyxai.diagnostics import RouteGraphBuilder


@pytest.fixture
def valid_route() -> dict:
    return {
        "route_id": "route:test",
        "nodes": [
            {
                "node_id": "preprocessor",
                "node_type": "preprocessing",
                "component_version": "v1",
                "registered_attributes": {"version": "v1", "schema": "s1"},
                "observed_attributes": {"version": "v1", "schema": "s1"},
                "mandatory": True,
                "repairable": True,
                "evidence_refs": ["evidence:preprocessor"],
            },
            {
                "node_id": "model",
                "node_type": "model",
                "component_version": "v4",
                "registered_attributes": {"version": "v4", "sha256": "abc"},
                "observed_attributes": {"version": "v4", "sha256": "abc"},
                "mandatory": True,
                "repairable": True,
                "evidence_refs": ["evidence:model"],
            },
        ],
        "edges": [
            {
                "edge_id": "preprocessor-to-model",
                "source": "preprocessor",
                "target": "model",
                "relation": "transforms",
                "mandatory": True,
                "registered_contract": {"compatible": True},
                "observed_contract": {"compatible": True},
                "repairable": True,
                "evidence_refs": ["evidence:edge"],
            }
        ],
    }


@pytest.fixture
def invalid_route(valid_route: dict) -> dict:
    route = {
        **valid_route,
        "nodes": [dict(item) for item in valid_route["nodes"]],
    }
    route["nodes"][0]["observed_attributes"] = {"version": "v2", "schema": "s2"}
    return route


@pytest.fixture
def valid_graph(valid_route: dict):
    return RouteGraphBuilder().build(valid_route)
