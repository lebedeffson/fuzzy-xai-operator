from __future__ import annotations

import math
from collections import defaultdict
from dataclasses import dataclass

from .graph import RepositoryEdge, RepositoryGraph, RepositoryNode

CONTRACTS = (
    "DEPENDENCY_VERSION",
    "DATA_CONTRACT",
    "SERIALIZATION",
    "ARTIFACT_PROVENANCE",
    "MODEL_LOADING",
    "CONFIGURATION",
)


@dataclass(frozen=True)
class AuditCandidate:
    node_id: str
    repository: str
    file_path: str | None
    symbol: str | None
    contract: str
    score: float
    confidence: float
    covered_obligations: tuple[str, ...]
    evidence_refs: tuple[str, ...]
    estimated_cost: float
    side_effect_risk: float


@dataclass(frozen=True)
class AuditResult:
    method: str
    status: str
    candidates: tuple[AuditCandidate, ...]
    selected_cut: tuple[str, ...]
    equivalent_cuts: tuple[tuple[str, ...], ...]
    covered_obligations: tuple[str, ...]
    uncovered_obligations: tuple[str, ...]
    coverage: float
    limitations: tuple[str, ...]


def _contract_scores(
    node: RepositoryNode,
    relations: dict[str, tuple[str, ...]],
    runtime_details: str,
) -> dict[str, float]:
    scores = {name: 0.0 for name in CONTRACTS}
    if node.kind == "dependency":
        scores["DEPENDENCY_VERSION"] += 5
    if node.kind == "configuration_key":
        scores["CONFIGURATION"] += 3
    if node.kind in {"data_schema"}:
        scores["DATA_CONTRACT"] += 5
    if node.kind in {"serialized_artifact"}:
        scores["SERIALIZATION"] += 5
    if node.kind in {"model_checkpoint"}:
        scores["MODEL_LOADING"] += 5
    suffix = (node.file_path or "").lower()
    semantic_tokens = {
        str(token).lower()
        for token in node.attributes.get("semantic_tokens", ())
    }
    if semantic_tokens & {"dtype", "shape", "columns", "fields", "schema", "ndim", "validate"}:
        scores["DATA_CONTRACT"] += 3
    if semantic_tokens & {
        "pickle",
        "load",
        "loads",
        "dump",
        "dumps",
        "decode",
        "encode",
        "serialize",
        "deserialize",
    }:
        scores["SERIALIZATION"] += 3
    if semantic_tokens & {"checkpoint", "state_dict", "load_model", "load_weights", "weights"}:
        scores["MODEL_LOADING"] += 3
    if semantic_tokens & {"cache", "metadata", "checksum", "digest", "artifact", "path"}:
        scores["ARTIFACT_PROVENANCE"] += 2
    if semantic_tokens & {"config", "configuration", "settings", "options"}:
        scores["CONFIGURATION"] += 2
    if suffix.endswith((".json", ".yaml", ".yml", ".toml", ".cfg", ".ini")):
        scores["CONFIGURATION"] += 1
        scores["ARTIFACT_PROVENANCE"] += 0.5
    for relation in relations.get(node.node_id, ()):
        if relation == "serializes":
            scores["SERIALIZATION"] += 3
        elif relation in {"reads", "writes", "loads"}:
            scores["ARTIFACT_PROVENANCE"] += 2
        elif relation == "configured_by":
            scores["CONFIGURATION"] += 2
        elif relation == "depends_on":
            scores["DEPENDENCY_VERSION"] += 2
    # Runtime tokens are secondary evidence. Structural node/edge evidence dominates.
    if any(
        token in runtime_details
        for token in (
            "shape",
            "dtype",
            "field",
            "column",
            "type",
            "array",
            "sequence",
            "validat",
        )
    ):
        scores["DATA_CONTRACT"] += 1
    if any(
        token in runtime_details
        for token in ("pickle", "decode", "json", "serialize", "deserialize")
    ):
        scores["SERIALIZATION"] += 1
    if any(token in runtime_details for token in ("checkpoint", "state_dict", "load_model")):
        scores["MODEL_LOADING"] += 1
    if any(token in runtime_details for token in ("version", "requirement", "dependency")):
        scores["DEPENDENCY_VERSION"] += 1
    return scores


def _causal_adjacency(
    graph: RepositoryGraph,
) -> dict[str, tuple[tuple[str, RepositoryEdge], ...]]:
    values: dict[str, list[tuple[str, RepositoryEdge]]] = defaultdict(list)
    reverse_relations = {
        "calls",
        "imports",
        "configured_by",
        "depends_on",
        "tested_by",
        "reads",
        "loads",
        "consumes",
    }
    forward_relations = {
        "fails_in",
        "produces",
        "consumes",
        "loads",
        "serializes",
        "writes",
        "explains",
    }
    for edge in graph.edges:
        if edge.relation in reverse_relations:
            values[edge.target].append((edge.source, edge))
        if edge.relation in forward_relations or edge.relation == "contains":
            values[edge.source].append((edge.target, edge))
    return {
        node_id: tuple(successors)
        for node_id, successors in values.items()
    }


def _coverage_index(
    graph: RepositoryGraph,
    adjacency: dict[str, tuple[tuple[str, RepositoryEdge], ...]],
    max_depth: int = 7,
) -> tuple[dict[str, tuple[str, ...]], dict[str, tuple[str, ...]]]:
    reverse: dict[str, list[tuple[str, RepositoryEdge]]] = defaultdict(list)
    for source, successors in adjacency.items():
        for target, edge in successors:
            reverse[target].append((source, edge))
    obligations: dict[str, set[str]] = defaultdict(set)
    refs: dict[str, set[str]] = defaultdict(set)
    for runtime in (node for node in graph.nodes if node.kind == "runtime_exception"):
        obligation = runtime.attributes.get("obligation")
        if not obligation:
            continue
        queue = [(runtime.node_id, 0)]
        visited = {runtime.node_id}
        while queue:
            current, depth = queue.pop(0)
            obligations[current].add(str(obligation))
            if depth >= max_depth:
                continue
            for predecessor, edge in reverse.get(current, ()):
                refs[predecessor].update(edge.evidence_refs)
                if predecessor in visited:
                    continue
                visited.add(predecessor)
                queue.append((predecessor, depth + 1))
    return (
        {node_id: tuple(sorted(values)) for node_id, values in obligations.items()},
        {node_id: tuple(sorted(values)) for node_id, values in refs.items()},
    )


def _fanout(
    adjacency: dict[str, tuple[tuple[str, RepositoryEdge], ...]],
    node_id: str,
) -> int:
    return len({successor for successor, _edge in adjacency.get(node_id, ())})


def _candidate(
    node: RepositoryNode,
    adjacency: dict[str, tuple[tuple[str, RepositoryEdge], ...]],
    coverage: dict[str, tuple[str, ...]],
    coverage_refs: dict[str, tuple[str, ...]],
    relations: dict[str, tuple[str, ...]],
    runtime_details: str,
    traceback_refs: frozenset[str],
) -> AuditCandidate | None:
    if node.kind in {
        "repository",
        "package",
        "module",
        "runtime_exception",
        "test",
        "fixture",
    }:
        return None
    normalized_path = (node.file_path or "").replace("\\", "/").lower()
    if node.kind == "file" and (
        normalized_path.startswith(("test/", "tests/"))
        or "/test_" in normalized_path
        or normalized_path.endswith("_test.py")
    ):
        return None
    if node.kind == "dependency":
        constraint = str(node.attributes.get("constraint", ""))
        if (
            not node.symbol
            or node.symbol.lower() not in runtime_details
            or not constraint
        ):
            return None
    covered = coverage.get(node.node_id, ())
    refs = tuple(sorted({*node.evidence_refs, *coverage_refs.get(node.node_id, ())}))
    if not covered:
        return None
    contract_scores = _contract_scores(node, relations, runtime_details)
    maximum_contract_score = max(contract_scores.values())
    contract = (
        min(
            name
            for name, score in contract_scores.items()
            if score == maximum_contract_score
        )
        if maximum_contract_score > 0
        else "INSUFFICIENT_EVIDENCE"
    )
    structural = 3.0 * len(covered)
    trace = sum(
        2.0
        for ref in refs
        if ref in traceback_refs
    )
    contract_strength = contract_scores.get(contract, 0.0)
    fanout = _fanout(adjacency, node.node_id)
    estimated_cost = 1.0 + math.log1p(fanout)
    risk = 0.05 * fanout
    score = structural + trace + contract_strength - 0.1 * estimated_cost
    confidence = 1.0 - math.exp(-max(0.0, score) / 8.0)
    return AuditCandidate(
        node.node_id,
        node.repository,
        node.file_path,
        node.symbol,
        contract,
        score,
        confidence,
        covered,
        refs,
        estimated_cost,
        risk,
    )


def _all_candidates(graph: RepositoryGraph) -> tuple[AuditCandidate, ...]:
    adjacency = _causal_adjacency(graph)
    coverage, coverage_refs = _coverage_index(graph, adjacency)
    relation_sets: dict[str, set[str]] = defaultdict(set)
    for edge in graph.edges:
        relation_sets[edge.source].add(edge.relation)
        relation_sets[edge.target].add(edge.relation)
    relations = {
        node_id: tuple(sorted(values))
        for node_id, values in relation_sets.items()
    }
    runtime_details = " ".join(
        item.detail.lower()
        for item in graph.evidence
        if item.kind in {"traceback", "failing_test"}
    )
    traceback_refs = frozenset(
        item.evidence_id
        for item in graph.evidence
        if item.kind == "traceback"
    )
    candidates = tuple(
        candidate
        for node in graph.nodes
        if (
            candidate := _candidate(
                node,
                adjacency,
                coverage,
                coverage_refs,
                relations,
                runtime_details,
                traceback_refs,
            )
        )
        is not None
    )
    return tuple(
        sorted(
            candidates,
            key=lambda item: (
                -item.score,
                -len(item.covered_obligations),
                item.estimated_cost,
                item.node_id,
            ),
        )
    )


def audit_greedy(graph: RepositoryGraph, *, abstention_threshold: float = 2.0) -> AuditResult:
    candidates = _all_candidates(graph)
    if not candidates or candidates[0].score < abstention_threshold:
        return AuditResult(
            "B_GREEDY",
            "INSUFFICIENT_EVIDENCE",
            candidates[:3],
            (),
            (),
            (),
            graph.obligations,
            0.0,
            (*graph.limitations, "calibrated_abstention"),
        )
    first_obligation = graph.obligations[0] if graph.obligations else ""
    selected = max(
        candidates,
        key=lambda item: (
            first_obligation in item.covered_obligations,
            -len(item.covered_obligations),
            item.score / max(1, len(item.covered_obligations)),
            -item.estimated_cost,
            item.node_id,
        ),
    )
    if selected.contract == "INSUFFICIENT_EVIDENCE":
        return AuditResult(
            "B_GREEDY",
            "INSUFFICIENT_EVIDENCE",
            candidates[:3],
            (),
            (),
            (),
            graph.obligations,
            0.0,
            (*graph.limitations, "contract_family_not_supported_by_evidence"),
        )
    covered = set(selected.covered_obligations)
    return AuditResult(
        "B_GREEDY",
        "DIAGNOSED",
        candidates[:3],
        (selected.node_id,),
        ((selected.node_id,),),
        tuple(sorted(covered)),
        tuple(sorted(set(graph.obligations) - covered)),
        len(covered) / max(1, len(graph.obligations)),
        graph.limitations,
    )


def _global_objective(
    cut: tuple[AuditCandidate, ...],
) -> tuple[float, int, tuple[str, ...]]:
    cost = sum(candidate.estimated_cost for candidate in cut)
    risk = sum(candidate.side_effect_risk for candidate in cut)
    objective = cost + 1.5 * risk
    return objective, len(cut), tuple(sorted(candidate.node_id for candidate in cut))


def _exact_cover(
    candidates: tuple[AuditCandidate, ...],
    obligations: tuple[str, ...],
) -> tuple[tuple[str, ...], tuple[tuple[str, ...], ...]] | None:
    if not obligations:
        return (), ((),)
    if len(obligations) > 20:
        return None
    bit = {obligation: 1 << index for index, obligation in enumerate(obligations)}
    full = (1 << len(obligations)) - 1
    states: dict[int, tuple[float, int, set[tuple[str, ...]]]] = {
        0: (0.0, 0, {()})
    }
    for candidate in candidates:
        candidate_mask = 0
        for obligation in candidate.covered_obligations:
            candidate_mask |= bit.get(obligation, 0)
        if not candidate_mask:
            continue
        candidate_value = candidate.estimated_cost + 1.5 * candidate.side_effect_risk
        updated = dict(states)
        for mask, (value, size, cuts) in states.items():
            new_mask = mask | candidate_mask
            new_value = value + candidate_value
            new_size = size + 1
            new_cuts = {
                tuple(sorted((*cut, candidate.node_id)))
                for cut in cuts
            }
            previous = updated.get(new_mask)
            better = previous is None or new_value < previous[0] - 1e-12
            if (
                previous is not None
                and abs(new_value - previous[0]) <= 1e-12
                and new_size < previous[1]
            ):
                better = True
            if better:
                updated[new_mask] = (new_value, new_size, new_cuts)
            elif (
                abs(new_value - previous[0]) <= 1e-12
                and new_size == previous[1]
            ):
                updated[new_mask] = (
                    previous[0],
                    previous[1],
                    {*previous[2], *new_cuts},
                )
        states = updated
    if full not in states:
        return None
    equivalent = tuple(sorted(states[full][2]))
    return min(equivalent), equivalent


def audit_global(graph: RepositoryGraph, *, abstention_threshold: float = 2.0) -> AuditResult:
    candidates = _all_candidates(graph)
    if not candidates:
        return AuditResult(
            "O_ROUTE",
            "INSUFFICIENT_EVIDENCE",
            (),
            (),
            (),
            (),
            graph.obligations,
            0.0,
            (*graph.limitations, "no_structural_candidate"),
        )
    obligations = frozenset(graph.obligations)
    exact = _exact_cover(candidates, graph.obligations)
    if exact is None:
        limitation = (
            "exact_cover_complexity_limit"
            if len(graph.obligations) > 20
            else "obligations_not_structurally_coverable"
        )
        return AuditResult(
            "O_ROUTE",
            "INSUFFICIENT_EVIDENCE",
            candidates[:3],
            (),
            (),
            (),
            graph.obligations,
            0.0,
            (*graph.limitations, limitation),
        )
    selected_ids, equivalent = exact
    selected = tuple(
        candidate
        for candidate in candidates
        if candidate.node_id in selected_ids
    )
    covered = {
        obligation
        for candidate in selected
        for obligation in candidate.covered_obligations
    }
    ranked = tuple(
        sorted(
            candidates,
            key=lambda item: (
                -len(set(item.covered_obligations) & covered),
                -item.score,
                item.estimated_cost,
                item.node_id,
            ),
        )
    )
    confidence = max((candidate.confidence for candidate in selected), default=0.0)
    if (
        confidence < 1.0 - math.exp(-abstention_threshold / 8.0)
        or any(candidate.contract == "INSUFFICIENT_EVIDENCE" for candidate in selected)
    ):
        return AuditResult(
            "O_ROUTE",
            "INSUFFICIENT_EVIDENCE",
            ranked[:3],
            (),
            (),
            (),
            graph.obligations,
            0.0,
            (
                *graph.limitations,
                (
                    "contract_family_not_supported_by_evidence"
                    if any(
                        candidate.contract == "INSUFFICIENT_EVIDENCE"
                        for candidate in selected
                    )
                    else "calibrated_abstention"
                ),
            ),
        )
    return AuditResult(
        "O_ROUTE",
        "DIAGNOSED",
        ranked[:3],
        tuple(sorted(candidate.node_id for candidate in selected)),
        tuple(sorted(equivalent)),
        tuple(sorted(covered)),
        tuple(sorted(obligations - covered)),
        len(covered) / max(1, len(obligations)),
        graph.limitations,
    )


class RepositoryRouteAuditor:
    def __init__(self, *, abstention_threshold: float = 2.0) -> None:
        self.abstention_threshold = abstention_threshold

    def audit(self, graph: RepositoryGraph, method: str = "O_ROUTE") -> AuditResult:
        if method == "O_ROUTE":
            return audit_global(graph, abstention_threshold=self.abstention_threshold)
        if method == "B_GREEDY":
            return audit_greedy(graph, abstention_threshold=self.abstention_threshold)
        raise ValueError(f"unsupported method: {method}")
