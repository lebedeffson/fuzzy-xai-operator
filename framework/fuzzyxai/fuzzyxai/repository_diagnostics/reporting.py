from __future__ import annotations

from dataclasses import dataclass

from .auditor import AuditResult
from .evidence_requests import EvidenceRequest
from .graph import RepositoryGraph
from .practical_recovery import VerifiableRepairPlan


@dataclass(frozen=True)
class DiagnosisSection:
    status: str
    probable_source: str
    contract: str
    evidence: tuple[str, ...]


@dataclass(frozen=True)
class NextStepSection:
    command: tuple[str, ...]
    expected_evidence: str


@dataclass(frozen=True)
class EngineeringDiagnosticReport:
    diagnosis: DiagnosisSection
    next_step: NextStepSection | None
    repair_plan: VerifiableRepairPlan | None
    audit_reference: str


def build_engineering_report(
    graph: RepositoryGraph,
    result: AuditResult,
    evidence_requests: tuple[EvidenceRequest, ...],
    repair_plan: VerifiableRepairPlan | None,
) -> EngineeringDiagnosticReport:
    candidate = result.candidates[0] if result.candidates else None
    evidence_index = {
        evidence.evidence_id: evidence.detail for evidence in graph.evidence
    }
    evidence = tuple(
        evidence_index[reference]
        for reference in (candidate.evidence_refs if candidate else ())
        if reference in evidence_index
    )
    source = (
        f"{candidate.file_path}::{candidate.symbol}"
        if candidate is not None
        else "Источник не локализован"
    )
    request = evidence_requests[0] if evidence_requests else None
    return EngineeringDiagnosticReport(
        DiagnosisSection(
            result.status,
            source,
            candidate.contract if candidate is not None else "UNREGISTERED_CONTRACT",
            evidence
            or (
                "Проверяемых свидетельств для окончательного диагноза недостаточно",
            ),
        ),
        (
            NextStepSection(request.command, request.expected_evidence)
            if request is not None
            else None
        ),
        repair_plan,
        graph.trace_sha256,
    )
