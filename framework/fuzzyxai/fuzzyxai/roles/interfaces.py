from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, runtime_checkable


@dataclass(frozen=True)
class RouteGraph:
    route_id: str
    trace_sha256: str


@dataclass(frozen=True)
class DiagnosticCut:
    element_ids: tuple[str, ...]
    cost: float


@dataclass(frozen=True)
class AuditReport:
    route_id: str
    issue_ids: tuple[str, ...]
    obligations: tuple[str, ...]


@dataclass(frozen=True)
class RepairPlan:
    action_ids: tuple[str, ...]


@dataclass(frozen=True)
class ExecutionReport:
    completed_action_ids: tuple[str, ...]
    new_critical_violations: tuple[str, ...]


@dataclass(frozen=True)
class RecertificationReport:
    status: str
    trace_sha256: str


@dataclass(frozen=True)
class CertifiedRoute:
    route_id: str
    trace_sha256: str


@dataclass(frozen=True)
class DomainDecision:
    code: str
    rationale: str


@runtime_checkable
class RouteAuditor(Protocol):
    def audit(self, route: RouteGraph) -> AuditReport: ...


@runtime_checkable
class RepairPlanner(Protocol):
    def build_plan(
        self,
        audit: AuditReport,
        cut: DiagnosticCut,
    ) -> RepairPlan: ...


@runtime_checkable
class RepairExecutor(Protocol):
    def execute(self, plan: RepairPlan) -> ExecutionReport: ...


@runtime_checkable
class RouteRecertifier(Protocol):
    def recertify(
        self,
        execution: ExecutionReport,
    ) -> RecertificationReport: ...


@runtime_checkable
class ExternalPolicy(Protocol):
    def decide(
        self,
        certified_route: CertifiedRoute,
    ) -> DomainDecision: ...
