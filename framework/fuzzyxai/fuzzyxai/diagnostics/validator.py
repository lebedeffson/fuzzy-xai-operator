from __future__ import annotations

from collections import defaultdict

from .causes import DiagnosticCauseAnalyzer
from .contract_registry import ContractCheck, ContractRegistry
from .contracts import (
    Contract,
    DiagnosticIssue,
    RouteGraph,
    ValidationObligation,
    ValidationResult,
)


def _atom(issue_id: str, subject: str, field: str | None, code: str) -> str:
    del issue_id
    location = f"field:{field}" if field else "subject"
    prefix = "edge" if subject.startswith("edge:") else "node"
    return f"{prefix}:{subject}/{location}/violation:{code}"


def _has_cycle(graph: RouteGraph) -> bool:
    adjacency: dict[str, list[str]] = defaultdict(list)
    for edge in graph.edges:
        if edge.mandatory:
            adjacency[edge.source].append(edge.target)
    visiting: set[str] = set()
    visited: set[str] = set()

    def visit(node: str) -> bool:
        if node in visiting:
            return True
        if node in visited:
            return False
        visiting.add(node)
        if any(visit(target) for target in adjacency[node]):
            return True
        visiting.remove(node)
        visited.add(node)
        return False

    return any(visit(node.node_id) for node in graph.nodes)


class DiagnosticValidator:
    def __init__(
        self,
        registry: ContractRegistry | None = None,
        cause_analyzer: DiagnosticCauseAnalyzer | None = None,
    ) -> None:
        self.registry = registry or ContractRegistry.default()
        self.cause_analyzer = cause_analyzer or DiagnosticCauseAnalyzer()

    def validate(self, graph: RouteGraph) -> ValidationResult:
        issues: list[DiagnosticIssue] = []
        obligations: list[ValidationObligation] = []
        passed: list[str] = []
        if not graph.nodes:
            empty = DiagnosticIssue(
                issue_id="issue:route:empty",
                category="provenance",
                code="insufficient_evidence",
                severity="error",
                symptom="Маршрут не содержит узлов и не может быть проверен.",
                violated_contract="route:non_empty",
                affected_nodes=(),
                affected_edges=(),
                affected_fields=(),
                source_nodes=(),
                cause_candidates=(),
                evidence_refs=(),
                confidence=None,
                repairable=False,
                insufficient_evidence=True,
                unknown=False,
            )
            empty = DiagnosticIssue(
                **{
                    **empty.__dict__,
                    "cause_candidates": self.cause_analyzer.analyze(graph, empty),
                }
            )
            issues.append(empty)
            obligations.append(ValidationObligation("route:non_empty", empty.issue_id, (), False))
        for contract in sorted(graph.contracts, key=lambda item: item.contract_id):
            check = self.registry.evaluate(contract, graph)
            if check.passed:
                passed.append(contract.contract_id)
                continue
            issue = self._issue(graph, contract, check)
            issue = DiagnosticIssue(
                **{
                    **issue.__dict__,
                    "cause_candidates": self.cause_analyzer.analyze(graph, issue),
                }
            )
            issues.append(issue)
            candidates = self._candidate_atoms(issue, contract)
            obligations.append(
                ValidationObligation(
                    obligation_id=contract.contract_id,
                    issue_id=issue.issue_id,
                    candidate_atoms=candidates,
                    repairable=contract.repairable,
                )
            )
        if _has_cycle(graph):
            issue_id = "issue:mandatory-route-cycle"
            issue = DiagnosticIssue(
                issue_id=issue_id,
                category="provenance",
                code="cyclic_mandatory_route",
                severity="error",
                symptom="В обязательном маршруте обнаружена циклическая зависимость.",
                violated_contract="route:acyclic",
                affected_nodes=tuple(node.node_id for node in graph.nodes),
                affected_edges=tuple(edge.edge_id for edge in graph.edges),
                affected_fields=(),
                source_nodes=(),
                cause_candidates=(),
                evidence_refs=(),
                confidence=1.0,
                repairable=False,
                insufficient_evidence=False,
                unknown=False,
            )
            issue = DiagnosticIssue(
                **{
                    **issue.__dict__,
                    "cause_candidates": self.cause_analyzer.analyze(graph, issue),
                }
            )
            issues.append(issue)
            obligations.append(ValidationObligation("route:acyclic", issue_id, (), False))
        status = self._status(tuple(issues))
        return ValidationResult(
            status=status,
            issues=tuple(issues),
            obligations=tuple(obligations),
            checked_contracts=tuple(contract.contract_id for contract in graph.contracts),
            passed_contracts=tuple(passed),
            graph_trace_sha256=graph.trace_sha256,
        )

    def _issue(self, graph: RouteGraph, contract: Contract, check: ContractCheck) -> DiagnosticIssue:
        node = graph.node(contract.subject_id)
        edge = graph.edge(contract.subject_id)
        actual = repr(check.actual)
        expected = repr(contract.expected)
        if check.insufficient_evidence:
            code = "insufficient_evidence"
            symptom = f"Контракт {contract.contract_id} не может быть проверен: {check.message}."
        elif contract.kind in {"checksum"}:
            code = "checksum_mismatch"
            symptom = f"Контрольная сумма {contract.subject_id}.{contract.field} не совпадает."
        elif contract.kind in {"equals", "compatible"}:
            code = "contract_mismatch"
            symptom = f"Для {contract.subject_id}.{contract.field} получено {actual}, требуется {expected}."
        else:
            code = f"contract_{contract.kind}_failed"
            symptom = f"Контракт {contract.contract_id} не выполнен: {check.message}."
        refs = tuple(dict.fromkeys((*contract.evidence_refs, *(node.evidence_refs if node else ()), *(edge.evidence_refs if edge else ()))))
        sources = contract.source_nodes or ((node.node_id,) if node else ((edge.source, edge.target) if edge else ()))
        return DiagnosticIssue(
            issue_id=f"issue:{contract.contract_id}",
            category=contract.category if contract.category else "unknown",
            code=code,
            severity=contract.severity,
            symptom=symptom,
            violated_contract=contract.contract_id,
            affected_nodes=(node.node_id,) if node else ((edge.source, edge.target) if edge else ()),
            affected_edges=(edge.edge_id,) if edge else (),
            affected_fields=(str(contract.field),) if contract.field else (),
            source_nodes=tuple(sources),
            cause_candidates=(),
            evidence_refs=refs,
            confidence=None if check.insufficient_evidence else 1.0,
            repairable=contract.repairable,
            insufficient_evidence=check.insufficient_evidence,
            unknown=contract.kind not in self.registry._evaluators,
        )

    @staticmethod
    def _candidate_atoms(issue: DiagnosticIssue, contract: Contract) -> tuple[str, ...]:
        if not contract.repairable:
            return ()
        subjects = tuple(dict.fromkeys((*issue.source_nodes, *issue.affected_edges, *issue.affected_nodes)))
        specific = tuple(_atom(issue.issue_id, subject, contract.field, issue.code) for subject in subjects)
        shared_sources = tuple(f"node:{source}/violation:source_component" for source in issue.source_nodes)
        return tuple(dict.fromkeys((*specific, *shared_sources)))

    @staticmethod
    def _status(issues: tuple[DiagnosticIssue, ...]) -> str:
        if not issues:
            return "valid"
        if all(issue.unknown for issue in issues):
            return "unknown"
        if all(issue.insufficient_evidence for issue in issues):
            return "insufficient_evidence"
        if any(issue.insufficient_evidence or issue.unknown for issue in issues):
            return "partially_valid"
        return "invalid"
