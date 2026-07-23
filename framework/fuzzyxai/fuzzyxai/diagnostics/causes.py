from __future__ import annotations

from .contracts import CauseStatement, DiagnosticIssue, RouteGraph


class DiagnosticCauseAnalyzer:
    """Build evidence-bound diagnostic chains without claiming an unobserved root cause."""

    def analyze(self, graph: RouteGraph, issue: DiagnosticIssue) -> tuple[CauseStatement, ...]:
        evidence = issue.evidence_refs
        statements = [
            CauseStatement(
                cause_id=f"{issue.issue_id}:symptom",
                level="symptom",
                statement=issue.symptom,
                supporting_evidence=evidence,
                confidence=issue.confidence,
                status="observed",
            ),
            CauseStatement(
                cause_id=f"{issue.issue_id}:contract",
                level="contract_violation",
                statement=f"Не выполнен контракт {issue.violated_contract}.",
                supporting_evidence=evidence,
                confidence=1.0,
                status="verified",
            ),
        ]
        if issue.unknown:
            statements.append(
                CauseStatement(
                    cause_id=f"{issue.issue_id}:proximate",
                    level="proximate_cause",
                    statement="Тип и ближайшая причина нарушения не установлены; требуется дополнительный контракт или журнал преобразования.",
                    supporting_evidence=evidence,
                    confidence=None,
                    status="insufficient_evidence",
                )
            )
        elif evidence and issue.source_nodes:
            statements.append(
                CauseStatement(
                    cause_id=f"{issue.issue_id}:proximate",
                    level="proximate_cause",
                    statement=f"Ближайшее установленное нарушение связано с {', '.join(issue.source_nodes)}.",
                    supporting_evidence=evidence,
                    confidence=issue.confidence,
                    status="evidence_supported",
                )
            )
            statements.append(
                CauseStatement(
                    cause_id=f"{issue.issue_id}:source",
                    level="source_component",
                    statement=f"Компонент-источник: {', '.join(issue.source_nodes)}.",
                    supporting_evidence=evidence,
                    confidence=issue.confidence,
                    status="localized",
                )
            )
        else:
            statements.append(
                CauseStatement(
                    cause_id=f"{issue.issue_id}:proximate",
                    level="proximate_cause",
                    statement="Ближайшая причина не установлена; требуются дополнительные сведения о маршруте.",
                    supporting_evidence=evidence,
                    confidence=None,
                    status="insufficient_evidence",
                )
            )
        return tuple(statements)
