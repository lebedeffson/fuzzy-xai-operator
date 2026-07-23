from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field as dataclass_field
from hashlib import sha256
from typing import Callable, Mapping


SCHEMA_VERSION = "1.0"
ROUTE_STATUSES = frozenset({"valid", "invalid", "insufficient_evidence", "unknown", "partially_valid"})


def canonical_json(value: object) -> bytes:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), default=str).encode("utf-8")


def canonical_sha256(value: object) -> str:
    return sha256(canonical_json(value)).hexdigest()


@dataclass(frozen=True)
class CauseStatement:
    cause_id: str
    level: str
    statement: str
    supporting_evidence: tuple[str, ...]
    confidence: float | None
    status: str


@dataclass(frozen=True)
class DiagnosticIssue:
    issue_id: str
    category: str
    code: str
    severity: str
    symptom: str
    violated_contract: str
    affected_nodes: tuple[str, ...]
    affected_edges: tuple[str, ...]
    affected_fields: tuple[str, ...]
    source_nodes: tuple[str, ...]
    cause_candidates: tuple[CauseStatement, ...]
    evidence_refs: tuple[str, ...]
    confidence: float | None
    repairable: bool
    insufficient_evidence: bool
    unknown: bool


@dataclass(frozen=True)
class DiagnosticCut:
    defect_atoms: tuple[str, ...]
    affected_nodes: tuple[str, ...]
    total_cost: float
    optimal: bool
    solver: str
    covered_obligations: tuple[str, ...]
    uncovered_obligations: tuple[str, ...]
    equivalent_optimal_cuts: tuple[tuple[str, ...], ...]
    runtime_ms: float


@dataclass(frozen=True)
class RepairStep:
    step_id: str
    title: str
    target: str
    provider_id: str
    operation: str
    parameters: dict[str, object]
    preconditions: tuple[str, ...]
    depends_on: tuple[str, ...]
    expected_postconditions: tuple[str, ...]
    verification_checks: tuple[str, ...]
    fallback_step_ids: tuple[str, ...]
    rollback_operation: str | None
    estimated_cost: float | None
    requires_human_approval: bool
    executable: bool


@dataclass(frozen=True)
class RepairPlan:
    plan_id: str
    cut: DiagnosticCut
    steps: tuple[RepairStep, ...]
    total_estimated_cost: float | None
    fully_executable: bool
    unresolved_issues: tuple[str, ...]
    trace_sha256: str


@dataclass(frozen=True)
class StepExecutionResult:
    step_id: str
    status: str
    changed: bool
    verification: tuple[dict[str, object], ...] = ()
    error: str | None = None


@dataclass(frozen=True)
class RecertificationReport:
    status: str
    route_valid_before: bool
    route_valid_after: bool
    completed_steps: tuple[str, ...]
    failed_steps: tuple[str, ...]
    resolved_issues: tuple[str, ...]
    remaining_issues: tuple[str, ...]
    new_issues: tuple[str, ...]
    verification_results: tuple[dict[str, object], ...]
    before_trace_sha256: str
    after_trace_sha256: str | None


@dataclass(frozen=True)
class DiagnosticReport:
    report_id: str
    route_id: str
    route_status: str
    issues: tuple[DiagnosticIssue, ...]
    minimal_cut: DiagnosticCut | None
    repair_plan: RepairPlan | None
    recertification: RecertificationReport | None
    user_summary: str
    expert_summary: str
    audit_summary: str
    limitations: tuple[str, ...]
    trace: bytes
    trace_sha256: str
    schema_version: str = SCHEMA_VERSION

    def summary(self, audience: str = "user") -> str:
        summaries = {
            "user": self.user_summary,
            "expert": self.expert_summary,
            "audit": self.audit_summary,
        }
        try:
            return summaries[audience]
        except KeyError as exc:
            raise ValueError("audience must be user, expert, or audit") from exc

    def to_dict(self) -> dict[str, object]:
        payload = asdict(self)
        payload["trace"] = self.trace.decode("utf-8")
        return payload


@dataclass(frozen=True)
class RouteNode:
    node_id: str
    node_type: str
    component_id: str
    component_version: str | None
    registered_attributes: dict[str, object]
    observed_attributes: dict[str, object]
    mandatory: bool
    repairable: bool
    evidence_refs: tuple[str, ...]


@dataclass(frozen=True)
class RouteEdge:
    edge_id: str
    source: str
    target: str
    relation: str
    mandatory: bool
    registered_contract: dict[str, object]
    observed_contract: dict[str, object]
    repairable: bool
    evidence_refs: tuple[str, ...] = ()


@dataclass(frozen=True)
class Contract:
    contract_id: str
    kind: str
    subject_id: str
    field: str | None = None
    expected: object | None = None
    severity: str = "error"
    category: str = "provenance"
    mandatory: bool = True
    repairable: bool = True
    evidence_refs: tuple[str, ...] = ()
    source_nodes: tuple[str, ...] = ()
    parameters: dict[str, object] = dataclass_field(default_factory=dict)


@dataclass(frozen=True)
class RouteGraph:
    route_id: str
    nodes: tuple[RouteNode, ...]
    edges: tuple[RouteEdge, ...]
    contracts: tuple[Contract, ...]
    metadata: dict[str, object] = dataclass_field(default_factory=dict)
    schema_version: str = SCHEMA_VERSION

    def node(self, node_id: str) -> RouteNode | None:
        return next((node for node in self.nodes if node.node_id == node_id), None)

    def edge(self, edge_id: str) -> RouteEdge | None:
        return next((edge for edge in self.edges if edge.edge_id == edge_id), None)

    def to_dict(self) -> dict[str, object]:
        return asdict(self)

    @property
    def trace_sha256(self) -> str:
        return canonical_sha256(self.to_dict())


@dataclass(frozen=True)
class ValidationObligation:
    obligation_id: str
    issue_id: str
    candidate_atoms: tuple[str, ...]
    repairable: bool


@dataclass(frozen=True)
class ValidationResult:
    status: str
    issues: tuple[DiagnosticIssue, ...]
    obligations: tuple[ValidationObligation, ...]
    checked_contracts: tuple[str, ...]
    passed_contracts: tuple[str, ...]
    graph_trace_sha256: str

    @property
    def valid(self) -> bool:
        return self.status == "valid"


@dataclass(frozen=True)
class RepairCostModel:
    costs: Mapping[str, float] = dataclass_field(default_factory=dict)
    default_cost: float = 1.0

    def cost(self, atom: str) -> float:
        value = float(self.costs.get(atom, self.default_cost))
        if value < 0:
            raise ValueError(f"repair cost cannot be negative: {atom}")
        return value


RepairHandler = Callable[[RouteGraph, RepairStep], RouteGraph]


@dataclass(frozen=True)
class RepairExecutionContext:
    handlers: Mapping[str, RepairHandler]
    approved_step_ids: frozenset[str] = frozenset()
    allow_external_changes: bool = False


@dataclass(frozen=True)
class BatchDiagnosticReport:
    reports: tuple[DiagnosticReport, ...]
    route_status_counts: dict[str, int]
    issue_category_counts: dict[str, int]
    frequent_source_nodes: tuple[tuple[str, int], ...]
    frequent_cuts: tuple[tuple[tuple[str, ...], int], ...]
    fully_executable_plan_rate: float
    unknown_issue_count: int
    runtime_ms: float
