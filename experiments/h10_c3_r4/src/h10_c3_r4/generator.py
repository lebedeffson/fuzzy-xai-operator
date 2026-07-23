from __future__ import annotations

import json
from collections import defaultdict
from dataclasses import replace
from hashlib import sha256
from pathlib import Path

from fuzzyxai.diagnostics import Contract, DiagnosticValidator, RouteEdge, RouteGraph, RouteNode

from .models import MutationRecord, R4Case, RouteTemplate


def _edge_role(source: str, target: str) -> str:
    return f"edge::{source}::{target}"


def _node_id(template: RouteTemplate, role: str) -> str:
    return f"node-{sha256(f'{template.template_id}:{role}'.encode()).hexdigest()[:12]}"


def _edge_id(template: RouteTemplate, source: str, target: str, index: int) -> str:
    token = f"{template.template_id}:{source}:{target}:{index}"
    return f"edge-{sha256(token.encode()).hexdigest()[:12]}"


def _contract_subject(
    role: str,
    node_ids: dict[str, str],
    edge_ids: dict[tuple[str, str], str],
) -> str:
    if role.startswith("edge::"):
        _, source, target = role.split("::", 2)
        return edge_ids[(source, target)]
    return node_ids[role]


def _atom_key(subject_id: str) -> str:
    return f"node:{subject_id}/field:_/violation:source_component"


def instantiate_valid_graph(template: RouteTemplate) -> RouteGraph:
    node_ids = {node.role: _node_id(template, node.role) for node in template.node_schema}
    edge_ids = {
        (edge.source_role, edge.target_role): _edge_id(
            template,
            edge.source_role,
            edge.target_role,
            index,
        )
        for index, edge in enumerate(template.edge_schema)
    }
    node_fields: dict[str, dict[str, object]] = defaultdict(dict)
    edge_fields: dict[str, dict[str, object]] = defaultdict(dict)
    for contract in template.contract_schema:
        subject_id = _contract_subject(contract.subject_role, node_ids, edge_ids)
        target = (
            edge_fields[subject_id]
            if contract.subject_role.startswith("edge::")
            else node_fields[subject_id]
        )
        target[contract.field] = contract.expected
    nodes = tuple(
        RouteNode(
            node_id=node_ids[node.role],
            node_type=node.node_type,
            component_id=f"{template.pipeline_family}:{node.role}",
            component_version="r4-registered",
            registered_attributes=dict(node_fields[node_ids[node.role]]),
            observed_attributes=dict(node_fields[node_ids[node.role]]),
            mandatory=True,
            repairable=True,
            evidence_refs=(
                f"template:{template.template_id}",
                f"node-role:{node.role}",
            ),
        )
        for node in template.node_schema
    )
    edges = tuple(
        RouteEdge(
            edge_id=edge_ids[(edge.source_role, edge.target_role)],
            source=node_ids[edge.source_role],
            target=node_ids[edge.target_role],
            relation=edge.relation,
            mandatory=edge.mandatory,
            registered_contract=dict(
                edge_fields[edge_ids[(edge.source_role, edge.target_role)]]
            ),
            observed_contract=dict(
                edge_fields[edge_ids[(edge.source_role, edge.target_role)]]
            ),
            repairable=True,
            evidence_refs=(
                f"template:{template.template_id}",
                f"edge-role:{edge.source_role}->{edge.target_role}",
            ),
            relation_status="known_valid",
        )
        for edge in template.edge_schema
    )
    contracts = tuple(
        Contract(
            contract_id=f"{template.template_id}:{contract.contract_id}",
            kind=contract.kind,
            subject_id=_contract_subject(
                contract.subject_role,
                node_ids,
                edge_ids,
            ),
            field=contract.field,
            expected=contract.expected,
            severity="error",
            category=contract.category,
            mandatory=True,
            repairable=contract.repairable,
            evidence_refs=(
                f"template:{template.template_id}",
                f"contract:{contract.contract_id}",
            ),
            source_nodes=tuple(node_ids[role] for role in contract.source_roles),
        )
        for contract in template.contract_schema
    )
    contract_by_short = {
        item.contract_id.rsplit(":", 1)[-1]: item for item in contracts
    }
    repair_costs: dict[str, float] = {}
    public_candidates = []
    for candidate in template.candidates:
        subject_id = node_ids[candidate.source_role]
        candidate_key = _atom_key(subject_id)
        repair_costs[candidate_key] = candidate.cost
        covered_contracts = tuple(
            contract_by_short[obligation]
            for obligation in candidate.covers
            if obligation in contract_by_short
        )
        for contract in covered_contracts:
            repair_costs[
                f"node:{subject_id}/field:{contract.field}/violation:contract_mismatch"
            ] = candidate.cost + 25.0
        public_candidates.append(
            {
                "candidate_id": candidate_key,
                "subject_id": subject_id,
                "covers": tuple(item.contract_id for item in covered_contracts),
                "cost": candidate.cost,
                "dependencies": tuple(
                    node_ids[
                        next(
                            item.source_role
                            for item in template.candidates
                            if item.candidate_id == dependency
                            or item.source_role == dependency
                        )
                    ]
                    for dependency in candidate.dependencies
                ),
                "executable": candidate.executable,
            }
        )
    for contract in contracts:
        code = "contract_mismatch"
        for node in nodes:
            repair_costs[
                f"node:{node.node_id}/field:{contract.field}/violation:{code}"
            ] = 50.0
        for edge in edges:
            repair_costs[
                f"edge:{edge.edge_id}/field:{contract.field}/violation:{code}"
            ] = 50.0
        repair_costs[
            f"contract:{contract.contract_id}/field:{contract.field}/violation:{code}"
        ] = 50.0
    repair_dependencies = {
        item["subject_id"]: tuple(item["dependencies"])
        for item in public_candidates
        if item["dependencies"]
    }
    return RouteGraph(
        route_id=f"route:{template.template_id}",
        nodes=nodes,
        edges=edges,
        contracts=contracts,
        metadata={
            "study_id": "FXAI-H10-C3-R4",
            "template_id": template.template_id,
            "template_hash": template.canonical_hash,
            "pipeline_family": template.pipeline_family,
            "modality": template.modality,
            "repair_costs": repair_costs,
            "public_candidates": public_candidates,
            "repair_dependencies": repair_dependencies,
        },
    )


def mutate_graph(
    template: RouteTemplate,
    graph: RouteGraph,
) -> tuple[RouteGraph, MutationRecord]:
    mutated_nodes = {node.node_id: node for node in graph.nodes}
    mutated_edges = {edge.edge_id: edge for edge in graph.edges}
    changed_nodes: set[str] = set()
    changed_edges: set[str] = set()
    for contract in graph.contracts:
        node = mutated_nodes.get(contract.subject_id)
        if node is not None:
            observed = dict(node.observed_attributes)
            observed[str(contract.field)] = f"mutated:{contract.contract_id}"
            mutated_nodes[node.node_id] = replace(
                node,
                observed_attributes=observed,
            )
            changed_nodes.add(node.node_id)
            continue
        edge = mutated_edges.get(contract.subject_id)
        if edge is not None:
            observed = dict(edge.observed_contract)
            observed[str(contract.field)] = f"mutated:{contract.contract_id}"
            mutated_edges[edge.edge_id] = replace(
                edge,
                observed_contract=observed,
            )
            changed_edges.add(edge.edge_id)
    mutated = replace(
        graph,
        nodes=tuple(mutated_nodes[node.node_id] for node in graph.nodes),
        edges=tuple(mutated_edges[edge.edge_id] for edge in graph.edges),
    )
    reverse_ids = tuple(
        item["candidate_id"]
        for item in graph.metadata["public_candidates"]
        if item["executable"]
    )
    mutation = MutationRecord(
        mutation_id=f"mutation:{template.template_id}",
        contract_ids=tuple(contract.contract_id for contract in graph.contracts),
        changed_nodes=tuple(sorted(changed_nodes)),
        changed_edges=tuple(sorted(changed_edges)),
        reverse_candidate_ids=tuple(sorted(reverse_ids)),
    )
    return mutated, mutation


def build_cases(
    templates: tuple[RouteTemplate, ...],
    *,
    cases_per_template: int = 1,
) -> tuple[R4Case, ...]:
    cases = []
    validator = DiagnosticValidator()
    for template in templates:
        valid = instantiate_valid_graph(template)
        if not validator.validate(valid).valid:
            raise AssertionError(f"template is not initially valid: {template.template_id}")
        mutated, mutation = mutate_graph(template, valid)
        if validator.validate(mutated).valid:
            raise AssertionError(f"mutation did not invalidate route: {template.template_id}")
        repairable = all(contract.repairable for contract in valid.contracts)
        for replicate in range(cases_per_template):
            cases.append(
                R4Case(
                    case_id=f"{template.template_id}:case:{replicate:03d}",
                    split=template.split,
                    pipeline_family=template.pipeline_family,
                    modality=template.modality,
                    stratum=template.stratum,
                    template_id=template.template_id,
                    template_hash=template.canonical_hash,
                    valid_graph=valid,
                    mutated_graph=mutated,
                    mutation=mutation,
                    repairable=repairable,
                )
            )
    return tuple(cases)


def write_cases(
    root: Path,
    split: str,
    cases: tuple[R4Case, ...],
    *,
    include_private: bool = True,
) -> dict[str, object]:
    public = root / "data" / split / "cases.jsonl"
    private = root / "private" / split / "mutation_log.jsonl"
    public.parent.mkdir(parents=True, exist_ok=True)
    public.write_text(
        "".join(
            json.dumps(case.public_view(), sort_keys=True) + "\n"
            for case in cases
        ),
        encoding="utf-8",
    )
    if include_private:
        private.parent.mkdir(parents=True, exist_ok=True)
        private.write_text(
            "".join(
                json.dumps(case.private_record(), sort_keys=True) + "\n"
                for case in cases
            ),
            encoding="utf-8",
        )
    manifest = {
        "split": split,
        "case_count": len(cases),
        "template_count": len({case.template_hash for case in cases}),
        "pipeline_families": len({case.pipeline_family for case in cases}),
        "private_mutation_log_exposed_to_methods": False,
        "private_mutation_log_stored": include_private,
        "template_hashes": sorted(
            {case.template_hash for case in cases}
        ),
        "case_hashes": [
            sha256(
                json.dumps(
                    case.public_view(),
                    sort_keys=True,
                    separators=(",", ":"),
                ).encode()
            ).hexdigest()
            for case in cases
        ],
    }
    (root / "data" / split / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return manifest
