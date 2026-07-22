"""Failure-path graph used by exact and approximate diagnostic cuts."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

from fuzzyxai.audit_certificate import ActionConditionedAuditCertificate


@dataclass(frozen=True)
class DiagnosticGraph:
    failure_paths: tuple[frozenset[str], ...]
    repair_costs: tuple[tuple[str, float], ...]
    fault_sources: tuple[tuple[str, str], ...]

    def __post_init__(self) -> None:
        if any(not path for path in self.failure_paths):
            raise ValueError("failure paths cannot be empty")

    @property
    def contracts(self) -> frozenset[str]:
        return frozenset().union(*self.failure_paths) if self.failure_paths else frozenset()

    def cost(self, contract_id: str) -> float:
        return dict(self.repair_costs).get(contract_id, 1.0)


def graph_from_certificate(
    certificate: ActionConditionedAuditCertificate,
    *,
    composite_paths: Iterable[Iterable[str]] = (),
) -> DiagnosticGraph:
    failed = [check for check in certificate.checks if not check.satisfied and check.requirement.blocking]
    paths = [frozenset((check.requirement.contract_id,)) for check in failed]
    known = {check.requirement.contract_id for check in failed}
    for composite in composite_paths:
        path = frozenset(item for item in composite if item in known)
        if path:
            paths.append(path)
    costs = tuple((check.requirement.contract_id, check.requirement.repair_cost) for check in failed)
    sources = tuple((check.requirement.contract_id, check.requirement.source_path) for check in failed)
    return DiagnosticGraph(tuple(dict.fromkeys(paths)), costs, sources)
