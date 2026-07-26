from __future__ import annotations

import math
from dataclasses import dataclass

from .auditor import AuditCandidate, AuditResult
from .contract_inference import (
    SUPPORTED_CONTRACTS,
    EvidenceGroundedContractInferer,
)
from .graph import RepositoryGraph
from .retrieval import EvidenceGroundedCandidateRetriever


@dataclass(frozen=True)
class CoverageRiskPoint:
    threshold: float
    coverage: float
    false_localization: float
    hit_at_3: float
    calibration_error: float


@dataclass(frozen=True)
class CalibrationObservation:
    confidence: float
    correct: bool
    hit_at_3: bool
    eligible: bool = True


def coverage_risk_curve(
    observations: tuple[CalibrationObservation, ...],
) -> tuple[CoverageRiskPoint, ...]:
    if not observations:
        return ()
    thresholds = tuple(
        sorted(
            {
                0.0,
                1.0,
                *(item.confidence for item in observations if item.eligible),
            }
        )
    )
    points = []
    for threshold in thresholds:
        accepted = [item for item in observations if item.eligible and item.confidence >= threshold]
        coverage = len(accepted) / len(observations)
        false_localization = sum(not item.correct for item in accepted) / len(observations)
        hit_at_3 = sum(item.hit_at_3 for item in accepted) / len(observations)
        calibration_error = sum(abs(item.confidence - float(item.correct)) for item in accepted) / max(1, len(accepted))
        points.append(
            CoverageRiskPoint(
                threshold,
                coverage,
                false_localization,
                hit_at_3,
                calibration_error,
            )
        )
    return tuple(points)


def select_abstention_threshold(
    observations: tuple[CalibrationObservation, ...],
    *,
    minimum_coverage: float = 0.70,
) -> CoverageRiskPoint:
    eligible = [point for point in coverage_risk_curve(observations) if point.coverage >= minimum_coverage]
    if not eligible:
        raise ValueError("no abstention threshold satisfies minimum coverage")
    return min(
        eligible,
        key=lambda point: (
            point.false_localization,
            -point.hit_at_3,
            point.calibration_error,
            -point.threshold,
        ),
    )


class EvidenceGroundedRouteAuditor:
    """Separate retrieval, contract inference and strategy-specific ranking."""

    def __init__(
        self,
        *,
        abstention_threshold: float = 0.35,
        candidate_threshold: float = 0.15,
        retriever: EvidenceGroundedCandidateRetriever | None = None,
        contract_inferer: EvidenceGroundedContractInferer | None = None,
    ) -> None:
        self.abstention_threshold = abstention_threshold
        self.candidate_threshold = min(
            candidate_threshold,
            abstention_threshold,
        )
        self.retriever = retriever or EvidenceGroundedCandidateRetriever()
        self.contract_inferer = contract_inferer or EvidenceGroundedContractInferer()

    def audit(self, graph: RepositoryGraph, method: str = "O_ROUTE") -> AuditResult:
        pool = self._candidate_pool(graph)
        if method == "B_GREEDY":
            return self._greedy(graph, pool)
        if method == "O_ROUTE":
            return self._global(graph, pool)
        raise ValueError(f"unsupported method: {method}")

    def _candidate_pool(
        self,
        graph: RepositoryGraph,
    ) -> tuple[AuditCandidate, ...]:
        fanout = {node.node_id: len({edge.target for edge in graph.edges if edge.source == node.node_id}) for node in graph.nodes}
        candidates = []
        for retrieved in self.retriever.retrieve(graph):
            inference = self.contract_inferer.infer(graph, retrieved)
            count = fanout.get(retrieved.node_id, 0)
            cost = 1.0 + math.log1p(count)
            risk = 0.05 * count
            confidence = 1.0 - (1.0 - retrieved.confidence) * (1.0 - inference.confidence) if inference.supported else 0.0
            candidates.append(
                AuditCandidate(
                    retrieved.node_id,
                    retrieved.repository,
                    retrieved.file_path,
                    retrieved.symbol,
                    inference.contract,
                    retrieved.retrieval_score + inference.score,
                    confidence,
                    retrieved.covered_obligations,
                    retrieved.evidence_refs,
                    cost,
                    risk,
                )
            )
        return tuple(
            sorted(
                candidates,
                key=lambda item: (
                    -item.score,
                    -item.confidence,
                    -len(item.covered_obligations),
                    item.estimated_cost,
                    item.node_id,
                ),
            )
        )

    def _greedy(
        self,
        graph: RepositoryGraph,
        pool: tuple[AuditCandidate, ...],
    ) -> AuditResult:
        ranked = pool
        supported = tuple(item for item in pool if item.contract in SUPPORTED_CONTRACTS and item.confidence >= self.abstention_threshold)
        if not supported:
            return self._candidate_or_abstain("B_GREEDY", graph, ranked)
        remaining = set(graph.obligations)
        selected: list[AuditCandidate] = []
        available = list(supported)
        while remaining and available:
            candidate = max(
                available,
                key=lambda item: (
                    len(remaining & set(item.covered_obligations)) / max(item.estimated_cost, 1e-12),
                    item.score,
                    item.confidence,
                    -item.estimated_cost,
                    item.node_id,
                ),
            )
            gain = remaining & set(candidate.covered_obligations)
            if not gain:
                break
            selected.append(candidate)
            remaining -= gain
            available.remove(candidate)
        if not selected:
            return self._abstain(
                "B_GREEDY",
                graph,
                ranked[:3],
                "obligations_not_structurally_coverable",
            )
        return self._diagnosed("B_GREEDY", graph, ranked[:3], tuple(selected))

    def _global(
        self,
        graph: RepositoryGraph,
        pool: tuple[AuditCandidate, ...],
    ) -> AuditResult:
        supported = tuple(item for item in pool if item.contract in SUPPORTED_CONTRACTS and item.confidence >= self.abstention_threshold)
        if not supported:
            return self._candidate_or_abstain("O_ROUTE", graph, pool)
        optimum = _optimal_cuts(supported, graph.obligations)
        if optimum is None:
            return self._abstain(
                "O_ROUTE",
                graph,
                self._global_ranking(supported, (), ())[:3],
                "obligations_not_structurally_coverable",
            )
        selected_ids, equivalent = optimum
        selected = tuple(item for item in supported if item.node_id in selected_ids)
        ranking = self._global_ranking(supported, selected_ids, equivalent)
        return self._diagnosed(
            "O_ROUTE",
            graph,
            ranking[:3],
            selected,
            equivalent,
        )

    @staticmethod
    def _global_ranking(
        pool: tuple[AuditCandidate, ...],
        selected_ids: tuple[str, ...],
        equivalent: tuple[tuple[str, ...], ...],
    ) -> tuple[AuditCandidate, ...]:
        selected = set(selected_ids)
        alternative = {node_id for cut in equivalent for node_id in cut} - selected
        return tuple(
            sorted(
                pool,
                key=lambda item: (
                    0 if item.node_id in selected else 1 if item.node_id in alternative else 2,
                    -len(item.covered_obligations),
                    item.estimated_cost + 1.5 * item.side_effect_risk,
                    -item.score,
                    item.node_id,
                ),
            )
        )

    @staticmethod
    def _candidate_result(
        method: str,
        graph: RepositoryGraph,
        candidates: tuple[AuditCandidate, ...],
    ) -> AuditResult:
        return AuditResult(
            method,
            "DIAGNOSIS_CANDIDATES",
            candidates[:3],
            (),
            (),
            (),
            graph.obligations,
            0.0,
            (*graph.limitations, "additional_evidence_required"),
        )

    def _candidate_or_abstain(
        self,
        method: str,
        graph: RepositoryGraph,
        pool: tuple[AuditCandidate, ...],
    ) -> AuditResult:
        eligible = tuple(item for item in pool if item.confidence >= self.candidate_threshold)
        if eligible:
            return self._candidate_result(method, graph, eligible)
        return self._abstain(
            method,
            graph,
            (),
            ("no_evidence_grounded_candidate" if not pool else "calibrated_abstention"),
        )

    @staticmethod
    def _abstain(
        method: str,
        graph: RepositoryGraph,
        candidates: tuple[AuditCandidate, ...],
        limitation: str,
    ) -> AuditResult:
        return AuditResult(
            method,
            "INSUFFICIENT_EVIDENCE",
            candidates,
            (),
            (),
            (),
            graph.obligations,
            0.0,
            (*graph.limitations, limitation),
        )

    @staticmethod
    def _diagnosed(
        method: str,
        graph: RepositoryGraph,
        candidates: tuple[AuditCandidate, ...],
        selected: tuple[AuditCandidate, ...],
        equivalent: tuple[tuple[str, ...], ...] | None = None,
    ) -> AuditResult:
        covered = {obligation for item in selected for obligation in item.covered_obligations}
        selected_ids = tuple(sorted(item.node_id for item in selected))
        return AuditResult(
            method,
            "DIAGNOSIS_CONFIRMED",
            candidates,
            selected_ids,
            equivalent or (selected_ids,),
            tuple(sorted(covered)),
            tuple(sorted(set(graph.obligations) - covered)),
            len(covered) / max(1, len(graph.obligations)),
            graph.limitations,
        )


def _optimal_cuts(
    candidates: tuple[AuditCandidate, ...],
    obligations: tuple[str, ...],
) -> tuple[tuple[str, ...], tuple[tuple[str, ...], ...]] | None:
    if not obligations:
        return (), ((),)
    if len(obligations) > 20:
        return None
    bits = {value: 1 << index for index, value in enumerate(obligations)}
    full = (1 << len(obligations)) - 1
    states: dict[int, tuple[float, int, set[tuple[str, ...]]]] = {0: (0.0, 0, {()})}
    for candidate in candidates:
        mask = 0
        for obligation in candidate.covered_obligations:
            mask |= bits.get(obligation, 0)
        if not mask:
            continue
        value = candidate.estimated_cost + 1.5 * candidate.side_effect_risk
        updated = dict(states)
        for current, (cost, size, cuts) in states.items():
            new_mask = current | mask
            new_cost = cost + value
            new_size = size + 1
            new_cuts = {tuple(sorted((*cut, candidate.node_id))) for cut in cuts}
            previous = updated.get(new_mask)
            if previous is None or new_cost < previous[0] - 1e-12 or (abs(new_cost - previous[0]) <= 1e-12 and new_size < previous[1]):
                updated[new_mask] = (new_cost, new_size, new_cuts)
            elif abs(new_cost - previous[0]) <= 1e-12 and new_size == previous[1]:
                updated[new_mask] = (
                    previous[0],
                    previous[1],
                    {*previous[2], *new_cuts},
                )
        states = updated
    if full not in states:
        return None
    equivalent = tuple(sorted(states[full][2]))
    return min(equivalent), equivalent
