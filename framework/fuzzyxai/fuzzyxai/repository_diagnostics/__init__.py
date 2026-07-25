"""Repository-grounded diagnostics built only from pre-fix observable evidence."""

from .auditor import (
    AuditCandidate,
    AuditResult,
    RepositoryRouteAuditor,
    audit_global,
    audit_greedy,
)
from .graph import (
    EvidenceRef,
    RepositoryEdge,
    RepositoryGraph,
    RepositoryNode,
)
from .importer import RepositoryIncident, RepositoryStructureImporter
from .recovery import (
    IncidentExecutionReport,
    IncidentSandboxExecutor,
    RegisteredRepair,
)

__all__ = [
    "AuditCandidate",
    "AuditResult",
    "EvidenceRef",
    "IncidentExecutionReport",
    "IncidentSandboxExecutor",
    "RegisteredRepair",
    "RepositoryEdge",
    "RepositoryGraph",
    "RepositoryIncident",
    "RepositoryNode",
    "RepositoryRouteAuditor",
    "RepositoryStructureImporter",
    "audit_global",
    "audit_greedy",
]
