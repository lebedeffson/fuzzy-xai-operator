from __future__ import annotations

from dataclasses import asdict, dataclass


@dataclass(frozen=True)
class Candidate:
    atom_id: str
    subject_kind: str
    subject_id: str
    field: str | None
    violation_code: str
    covers: tuple[str, ...]
    cost: float
    repairable: bool = True
    executable: bool = True
    dependencies: tuple[str, ...] = ()
    alternatives: tuple[str, ...] = ()
    provider_status: str = "healthy"
    conflicts: tuple[str, ...] = ()


@dataclass(frozen=True)
class Mutation:
    operation_id: str
    changed_nodes: tuple[str, ...]
    changed_edges: tuple[str, ...]
    broken_obligations: tuple[str, ...]
    allowed_inverse_ids: tuple[str, ...]


@dataclass(frozen=True)
class Case:
    case_id: str
    split: str
    pipeline: str
    modality: str
    stratum: str
    family: str
    obligations: tuple[str, ...]
    nodes: tuple[str, ...]
    edges: tuple[tuple[str, str, str], ...]
    candidates: tuple[Candidate, ...]
    mutations: tuple[Mutation, ...]
    repairable: bool

    def method_view(self) -> dict[str, object]:
        return {
            "case_id": self.case_id,
            "pipeline": self.pipeline,
            "modality": self.modality,
            "stratum": self.stratum,
            "obligations": list(self.obligations),
            "nodes": list(self.nodes),
            "edges": [list(edge) for edge in self.edges],
            "candidates": [asdict(candidate) for candidate in self.candidates],
        }

    def private_record(self) -> dict[str, object]:
        return {
            "case_id": self.case_id,
            "family": self.family,
            "mutations": [asdict(item) for item in self.mutations],
            "repairable": self.repairable,
        }


@dataclass(frozen=True)
class Gold:
    case_id: str
    status: str
    optimal_cuts: tuple[tuple[str, ...], ...]
    optimal_cost: float | None
    repairable: bool
    solver_a_cost: float | None
    solver_b_cost: float | None


@dataclass(frozen=True)
class MethodResult:
    method: str
    cut: tuple[str, ...]
    plan: tuple[str, ...]
    predicted_cost: float
    runtime_ms: float
    status: str

