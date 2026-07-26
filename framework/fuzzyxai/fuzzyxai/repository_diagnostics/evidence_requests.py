from __future__ import annotations

from dataclasses import dataclass

from .auditor import AuditResult
from .graph import RepositoryGraph


@dataclass(frozen=True)
class EvidenceRequest:
    command: tuple[str, ...]
    expected_evidence: str
    affected_candidates: tuple[str, ...]
    estimated_cost: float
    safety_level: str


class EvidenceRequestPlanner:
    """Propose one read-only diagnostic step; never execute it implicitly."""

    def plan(
        self,
        graph: RepositoryGraph,
        result: AuditResult,
    ) -> tuple[EvidenceRequest, ...]:
        if result.status == "DIAGNOSIS_CONFIRMED":
            return ()
        tests = tuple(str(node.symbol) for node in graph.nodes if node.kind == "runtime_exception" and node.symbol)
        if not tests:
            return ()
        candidates = tuple(item.node_id for item in result.candidates)
        expected = "runtime argument types and local values at candidate frames" if candidates else "exact project traceback frames and assertion difference"
        return (
            EvidenceRequest(
                (
                    "python",
                    "-m",
                    "pytest",
                    tests[0],
                    "-x",
                    "-vv",
                    "--showlocals",
                ),
                expected,
                candidates,
                1.0,
                "READ_ONLY_TEST_EXECUTION",
            ),
        )
