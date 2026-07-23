from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Literal


@dataclass(frozen=True)
class NodeTemplate:
    role: str
    node_type: str
    attributes: tuple[str, ...]


@dataclass(frozen=True)
class EdgeTemplate:
    source_role: str
    target_role: str
    relation: str
    mandatory: bool = True


@dataclass(frozen=True)
class ContractTemplate:
    contract_id: str
    subject_role: str
    kind: str
    field: str
    expected: str
    category: str
    source_roles: tuple[str, ...]
    repairable: bool = True


@dataclass(frozen=True)
class RepairCandidateTemplate:
    candidate_id: str
    source_role: str
    covers: tuple[str, ...]
    cost: float
    dependencies: tuple[str, ...] = ()
    executable: bool = True


@dataclass(frozen=True)
class RouteTemplate:
    template_id: str
    split: str
    pipeline_family: str
    modality: str
    stratum: Literal["S2", "S3", "S4", "S5"]
    node_schema: tuple[NodeTemplate, ...]
    edge_schema: tuple[EdgeTemplate, ...]
    contract_schema: tuple[ContractTemplate, ...]
    candidates: tuple[RepairCandidateTemplate, ...]
    mutation_grammar_id: str
    repair_grammar_id: str
    graph_hash: str
    coverage_hash: str
    mutation_hash: str
    repair_dependency_hash: str
    cost_hash: str
    canonical_hash: str

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True)
class MutationRecord:
    mutation_id: str
    contract_ids: tuple[str, ...]
    changed_nodes: tuple[str, ...]
    changed_edges: tuple[str, ...]
    reverse_candidate_ids: tuple[str, ...]


@dataclass(frozen=True)
class R4Case:
    case_id: str
    split: str
    pipeline_family: str
    modality: str
    stratum: str
    template_id: str
    template_hash: str
    valid_graph: object
    mutated_graph: object
    mutation: MutationRecord
    repairable: bool

    def public_view(self) -> dict[str, object]:
        return {
            "case_id": self.case_id,
            "pipeline_family": self.pipeline_family,
            "modality": self.modality,
            "stratum": self.stratum,
            "template_id": self.template_id,
            "template_hash": self.template_hash,
            "route_graph": self.mutated_graph.to_dict(),
        }

    def private_record(self) -> dict[str, object]:
        return {
            "case_id": self.case_id,
            "mutation": asdict(self.mutation),
            "repairable": self.repairable,
            "valid_graph_sha256": self.valid_graph.trace_sha256,
        }


@dataclass(frozen=True)
class PublicCandidate:
    candidate_id: str
    subject_kind: str
    subject_id: str
    field: str | None
    violation_code: str
    covers: tuple[str, ...]
    cost: float


@dataclass(frozen=True)
class R4MethodResult:
    method: str
    cut: tuple[str, ...]
    predicted_cost: float
    status: str
    runtime_ms: float


@dataclass(frozen=True)
class R4Gold:
    case_id: str
    status: str
    optimal_cuts: tuple[tuple[str, ...], ...]
    optimal_cost: float | None
    solver_a_cost: float | None
    solver_b_cost: float | None
    repairable: bool
