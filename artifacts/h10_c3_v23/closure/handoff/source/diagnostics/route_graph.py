from __future__ import annotations

from collections.abc import Mapping
from dataclasses import asdict, is_dataclass
from fuzzyxai.core.types import OperatorEdge, OperatorNode, OperatorRoute, ProofTrace

from .contracts import Contract, RouteEdge, RouteGraph, RouteNode


RELATIONS = frozenset(
    {
        "produces",
        "consumes",
        "derived_from",
        "explains",
        "calibrates",
        "transforms",
        "validates",
        "aggregates",
        "reduces",
        "certifies",
        "blocks",
    }
)


def _mapping(value: object) -> dict[str, object]:
    if isinstance(value, Mapping):
        return dict(value)
    if is_dataclass(value):
        return asdict(value)
    raise TypeError(f"cannot convert {type(value).__name__} to a route mapping")


def _node_from_operator(node: OperatorNode) -> RouteNode:
    raw = dict(node.raw)
    details = dict(node.details)
    registered = dict(raw.get("registered_attributes", {}))
    observed = dict(raw.get("observed_attributes", {}))
    if not observed:
        observed = {
            key: value
            for key, value in {
                **raw,
                "status": node.status,
                "value_source": node.value_source,
            }.items()
            if key not in {"registered_attributes", "observed_attributes"}
        }
    refs = tuple(
        dict.fromkeys(
            str(ref)
            for ref in (
                node.trace_ref,
                *node.input_refs,
                *node.output_refs,
                *details.get("evidence_refs", ()),
            )
            if ref
        )
    )
    return RouteNode(
        node_id=node.node_id,
        node_type=node.operator_type or details.get("node_type", "operator"),
        component_id=str(details.get("component_id", node.node_id)),
        component_version=details.get("component_version") or raw.get("version"),
        registered_attributes=registered,
        observed_attributes=observed,
        mandatory=bool(details.get("mandatory", True)),
        repairable=bool(details.get("repairable", True)),
        evidence_refs=refs,
    )


def _edge_from_operator(edge: OperatorEdge) -> RouteEdge:
    values = dict(edge.passed_values)
    raw_relation = values.pop("relation", None)
    if raw_relation is None:
        relation = "unknown_relation"
        relation_status = "insufficient_evidence"
    else:
        relation = str(raw_relation)
        relation_status = "known_valid" if relation in RELATIONS else "unsupported_relation"
    return RouteEdge(
        edge_id=edge.edge_id,
        source=edge.source_node_id,
        target=edge.target_node_id,
        relation=relation,
        mandatory=bool(values.pop("mandatory", True)),
        registered_contract=dict(values.pop("registered_contract", {})),
        observed_contract=dict(values.pop("observed_contract", values)),
        repairable=bool(values.pop("repairable", True)),
        evidence_refs=tuple(values.pop("evidence_refs", ())),
        relation_status=relation_status,
    )


class RouteGraphBuilder:
    """Convert public route objects into one immutable diagnostic graph."""

    def build(self, route: object) -> RouteGraph:
        if isinstance(route, RouteGraph):
            return route
        if isinstance(route, ProofTrace):
            payload = dict(route.route)
            payload.setdefault("route_id", f"proof:{route.scenario_id}")
            payload.setdefault("metadata", {})["proof_schema_version"] = route.schema_version
            return self._from_mapping(payload)
        if isinstance(route, OperatorRoute):
            return self._from_operator_route(route)
        if route.__class__.__name__ == "RouteObservation":
            return self._from_observation(_mapping(route))
        if hasattr(route, "explanation_graph") and not isinstance(route, Mapping):
            return self.build(getattr(route, "explanation_graph"))
        if isinstance(route, Mapping):
            return self._from_mapping(dict(route))
        raise TypeError(f"unsupported route type: {type(route).__name__}")

    def _from_operator_route(self, route: OperatorRoute) -> RouteGraph:
        nodes = tuple(_node_from_operator(node) for node in route.nodes)
        edges = tuple(_edge_from_operator(edge) for edge in route.edges)
        contracts = self._contracts_from_nodes_and_edges(nodes, edges)
        return RouteGraph(
            route_id=route.route_id or route.scenario_id,
            nodes=nodes,
            edges=edges,
            contracts=contracts,
            metadata={
                "source_type": "OperatorRoute",
                "scenario_id": route.scenario_id,
                "source_commit": route.source_commit,
            },
        )

    def _from_observation(self, payload: dict[str, object]) -> RouteGraph:
        expected = dict(payload.get("expected", {}))
        observed = dict(payload.get("observed", {}))
        mandatory = set(payload.get("mandatory_fields", ()))
        costs = dict(payload.get("repair_costs", {}))
        node_ids = tuple(dict.fromkeys((*expected, *observed, *mandatory)))
        nodes = tuple(
            RouteNode(
                node_id=str(field),
                node_type="route_field",
                component_id=str(field),
                component_version=None,
                registered_attributes={"value": expected.get(field)} if field in expected else {},
                observed_attributes={"value": observed.get(field)} if field in observed else {},
                mandatory=field in mandatory,
                repairable=True,
                evidence_refs=(f"route:{payload.get('route_id', 'observation')}/field:{field}",),
            )
            for field in node_ids
        )
        edge_pairs: list[tuple[str, str]] = []
        for path in payload.get("dependency_paths", ()):
            edge_pairs.extend((str(left), str(right)) for left, right in zip(path, path[1:]))
        edges = tuple(
            RouteEdge(
                edge_id=f"edge:{source}->{target}",
                source=source,
                target=target,
                relation="derived_from",
                mandatory=True,
                registered_contract={"compatible": True},
                observed_contract={"compatible": True},
                repairable=True,
                relation_status="known_valid",
            )
            for source, target in dict.fromkeys(edge_pairs)
        )
        contracts = self._contracts_from_nodes_and_edges(nodes, edges)
        return RouteGraph(
            route_id=str(payload.get("route_id", "route-observation")),
            nodes=nodes,
            edges=edges,
            contracts=contracts,
            metadata={
                "source_type": "RouteObservation",
                "dataset_id": payload.get("dataset_id"),
                "modality": payload.get("modality"),
                "object_id": payload.get("object_id"),
                "repair_costs": costs,
            },
        )

    def _from_mapping(self, payload: dict[str, object]) -> RouteGraph:
        if "route" in payload and isinstance(payload["route"], Mapping):
            return self._from_mapping(dict(payload["route"]))
        nodes = tuple(self._node_from_mapping(item) for item in payload.get("nodes", ()))
        edges = tuple(self._edge_from_mapping(item) for item in payload.get("edges", ()))
        explicit = tuple(self._contract_from_mapping(item) for item in payload.get("contracts", ()))
        contracts = explicit or self._contracts_from_nodes_and_edges(nodes, edges)
        return RouteGraph(
            route_id=str(payload.get("route_id") or payload.get("scenario_id") or "route"),
            nodes=nodes,
            edges=edges,
            contracts=contracts,
            metadata=dict(payload.get("metadata", {})),
            schema_version=str(payload.get("schema_version", "1.0")),
        )

    @staticmethod
    def _node_from_mapping(value: object) -> RouteNode:
        data = _mapping(value)
        node_id = str(data.get("node_id") or data.get("id"))
        details = dict(data.get("details", {}))
        raw = dict(data.get("raw", {}))
        registered = dict(data.get("registered_attributes", raw.get("registered_attributes", {})))
        observed = dict(data.get("observed_attributes", raw.get("observed_attributes", {})))
        if not observed:
            observed = dict(data.get("attributes", {}))
        return RouteNode(
            node_id=node_id,
            node_type=str(data.get("node_type") or data.get("kind") or data.get("operator_type") or "component"),
            component_id=str(data.get("component_id") or details.get("component_id") or node_id),
            component_version=data.get("component_version") or details.get("component_version"),
            registered_attributes=registered,
            observed_attributes=observed,
            mandatory=bool(data.get("mandatory", details.get("mandatory", True))),
            repairable=bool(data.get("repairable", details.get("repairable", True))),
            evidence_refs=tuple(str(item) for item in data.get("evidence_refs", details.get("evidence_refs", ()))),
        )

    @staticmethod
    def _edge_from_mapping(value: object) -> RouteEdge:
        data = _mapping(value)
        source = str(data.get("source") or data.get("source_node_id"))
        target = str(data.get("target") or data.get("target_node_id"))
        raw_relation = data.get("relation")
        if raw_relation is None:
            relation = "unknown_relation"
            relation_status = "insufficient_evidence"
        else:
            relation = str(raw_relation)
            relation_status = str(
                data.get(
                    "relation_status",
                    "known_valid" if relation in RELATIONS else "unsupported_relation",
                )
            )
        return RouteEdge(
            edge_id=str(data.get("edge_id") or f"edge:{source}->{target}"),
            source=source,
            target=target,
            relation=relation,
            mandatory=bool(data.get("mandatory", True)),
            registered_contract=dict(data.get("registered_contract", {})),
            observed_contract=dict(data.get("observed_contract", {})),
            repairable=bool(data.get("repairable", True)),
            evidence_refs=tuple(str(item) for item in data.get("evidence_refs", ())),
            relation_status=relation_status,
        )

    @staticmethod
    def _contract_from_mapping(value: object) -> Contract:
        data = _mapping(value)
        return Contract(
            contract_id=str(data["contract_id"]),
            kind=str(data["kind"]),
            subject_id=str(data["subject_id"]),
            field=data.get("field"),
            expected=data.get("expected"),
            severity=str(data.get("severity", "error")),
            category=str(data.get("category", "provenance")),
            mandatory=bool(data.get("mandatory", True)),
            repairable=bool(data.get("repairable", True)),
            evidence_refs=tuple(str(item) for item in data.get("evidence_refs", ())),
            source_nodes=tuple(str(item) for item in data.get("source_nodes", ())),
            parameters=dict(data.get("parameters", {})),
        )

    @staticmethod
    def _contracts_from_nodes_and_edges(
        nodes: tuple[RouteNode, ...],
        edges: tuple[RouteEdge, ...],
    ) -> tuple[Contract, ...]:
        contracts: list[Contract] = []
        for node in nodes:
            if node.mandatory:
                contracts.append(
                    Contract(
                        contract_id=f"node:{node.node_id}:present",
                        kind="node_present",
                        subject_id=node.node_id,
                        category=node.node_type if node.node_type in {"data", "preprocessing", "model", "calibration", "explainer", "rule", "provenance", "representation", "reduction", "serialization", "runtime"} else "provenance",
                        repairable=node.repairable,
                        evidence_refs=node.evidence_refs,
                        source_nodes=(node.node_id,),
                    )
                )
            for field, expected in sorted(node.registered_attributes.items()):
                kind = "required_attribute" if expected is None else "equals"
                contracts.append(
                    Contract(
                        contract_id=f"node:{node.node_id}:field:{field}",
                        kind=kind,
                        subject_id=node.node_id,
                        field=str(field),
                        expected=expected,
                        category=node.node_type if node.node_type in {"data", "preprocessing", "model", "calibration", "explainer", "rule", "provenance", "representation", "reduction", "serialization", "runtime"} else "provenance",
                        repairable=node.repairable,
                        evidence_refs=node.evidence_refs,
                        source_nodes=(node.node_id,),
                    )
                )
        for edge in edges:
            contracts.append(
                Contract(
                    contract_id=f"edge:{edge.edge_id}:relation",
                    kind="relation_known",
                    subject_id=edge.edge_id,
                    field="relation",
                    expected="known_valid",
                    category="provenance",
                    repairable=edge.repairable,
                    evidence_refs=edge.evidence_refs,
                    source_nodes=(edge.source, edge.target),
                )
            )
            if edge.mandatory:
                contracts.append(
                    Contract(
                        contract_id=f"edge:{edge.edge_id}:present",
                        kind="edge_present",
                        subject_id=edge.edge_id,
                        category="provenance",
                        repairable=edge.repairable,
                        evidence_refs=edge.evidence_refs,
                        source_nodes=(edge.source, edge.target),
                    )
                )
            for field, expected in sorted(edge.registered_contract.items()):
                contracts.append(
                    Contract(
                        contract_id=f"edge:{edge.edge_id}:field:{field}",
                        kind="equals",
                        subject_id=edge.edge_id,
                        field=str(field),
                        expected=expected,
                        category="provenance",
                        repairable=edge.repairable,
                        evidence_refs=edge.evidence_refs,
                        source_nodes=(edge.source, edge.target),
                    )
                )
        return tuple(contracts)
