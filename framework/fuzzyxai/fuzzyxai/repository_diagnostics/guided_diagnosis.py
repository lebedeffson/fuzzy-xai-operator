from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass, replace
from typing import Protocol

from .active_evidence import (
    ActiveEvidenceRequestPlanner,
    EvidenceRequest,
    R10TargetedProbePlanner,
    apply_probe_observation,
)
from .contract_inference_v2 import (
    ContractPrediction,
    HierarchicalContractInferenceEngine,
)
from .graph import RepositoryGraph
from .guided_retrieval import (
    R9_SOURCE_KINDS,
    BM25Retriever,
    CandidateReservoir,
    DenseCodeEncoder,
    DenseRetriever,
    ExactSymbolExtractor,
    FileRetriever,
    IncidentNormalizer,
    IncidentQuery,
    R9CandidateCompressor,
    R9CompressionConfig,
    R10RetrievalConfig,
    R10SourceAwareReranker,
    R10SymbolPoolBuilder,
    RankedSymbol,
    RepoGraphRanker,
    StrictIdentifierExtractor,
    StructuralReranker,
    SymbolDocument,
    documents_from_graph,
    reciprocal_rank_fusion,
)
from .incident_router import IncidentRouter, RoutingDecision
from .retrieval import EvidenceGroundedCandidateRetriever
from .runtime_events import RuntimeEvent

VARIANTS = tuple(f"R{index}" for index in range(9))
R9_VARIANTS = ("R9A", "R9B", "R9C")
R10_VARIANTS = ("R10A", "R10B", "R10C", "R10D")
BASELINES = (
    "B_TRACE",
    "B_BM25",
    "B_DENSE",
    "B_RRF",
    "B_GREEDY",
    "B_REPOGRAPH",
    "B_AGENTLESS_LOC",
    "O_ROUTE",
)
METHODS = (*VARIANTS, *R9_VARIANTS, *R10_VARIANTS, *BASELINES)


class CrossEncoder(Protocol):
    model_name: str
    revision: str

    def score(
        self,
        query: str,
        documents: Sequence[SymbolDocument],
    ) -> Sequence[float]:
        ...


class LocalTransformerCrossEncoder:
    """Pinned local-only pair scorer with no download fallback."""

    def __init__(self, model_name: str, revision: str) -> None:
        self.model_name = model_name
        self.revision = revision

    def score(
        self,
        query: str,
        documents: Sequence[SymbolDocument],
    ) -> Sequence[float]:
        try:
            import torch
            from transformers import (
                AutoModelForSequenceClassification,
                AutoTokenizer,
            )
        except ImportError as exc:
            raise RuntimeError("transformers backend is not installed") from exc
        tokenizer = AutoTokenizer.from_pretrained(
            self.model_name,
            revision=self.revision,
            local_files_only=True,
        )
        model = AutoModelForSequenceClassification.from_pretrained(
            self.model_name,
            revision=self.revision,
            local_files_only=True,
        )
        model.eval()
        values = []
        with torch.no_grad():
            for document in documents:
                inputs = tokenizer(
                    query,
                    document.text,
                    truncation=True,
                    max_length=384,
                    return_tensors="pt",
                )
                logits = model(**inputs).logits[0]
                values.append(float(logits[-1]))
        return values


@dataclass(frozen=True)
class GuidedCandidate:
    node_id: str
    file_path: str
    symbol: str | None
    score: float
    contract: ContractPrediction
    contract_hypotheses: tuple[ContractPrediction, ...]
    rank_sources: tuple[str, ...]
    line_count: int
    obligations: tuple[str, ...]
    evidence: tuple[str, ...] = ()


@dataclass(frozen=True)
class ExplorerAction:
    action: str
    target: str
    observation: str


@dataclass(frozen=True)
class GuidedDiagnosis:
    variant: str
    status: str
    candidates: tuple[GuidedCandidate, ...]
    evidence_requests: tuple[EvidenceRequest, ...]
    route: RoutingDecision
    trajectory: tuple[ExplorerAction, ...] = ()
    unavailable_reason: str | None = None
    active_evidence_status: str = "NOT_REQUESTED"
    active_evidence_details: tuple[tuple[str, str], ...] = ()


class EvidenceCalibrator:
    """Fail closed unless independent observations support the diagnosis."""

    @staticmethod
    def is_confirmed(candidate: GuidedCandidate) -> bool:
        if candidate.contract.family == "UNKNOWN_CONTRACT":
            return False
        direct_observations = sum(
            evidence.startswith("direct_observation:")
            for evidence in candidate.contract.evidence
        )
        return direct_observations >= 2


class BoundedRepositoryExplorer:
    ALLOWED_ACTIONS = frozenset(
        {
            "inspect_assertion",
            "inspect_configuration",
            "inspect_manifest",
            "inspect_traceback_frame",
            "request_runtime_value",
            "run_safe_probe",
            "search_symbol",
            "show_callees",
            "show_callers",
            "show_definition",
            "show_references",
        }
    )

    def __init__(self, action_budget: int = 12) -> None:
        if not 1 <= action_budget <= 12:
            raise ValueError("bounded explorer action budget must be in [1, 12]")
        self.action_budget = action_budget

    def explore(
        self,
        query: IncidentQuery,
        ranking: Sequence[RankedSymbol],
    ) -> tuple[tuple[RankedSymbol, ...], tuple[ExplorerAction, ...]]:
        actions: list[ExplorerAction] = []
        for candidate in ranking[: self.action_budget]:
            action = (
                "inspect_traceback_frame"
                if "traceback" in candidate.rank_sources
                else "show_definition"
            )
            if action not in self.ALLOWED_ACTIONS:
                raise AssertionError(action)
            actions.append(
                ExplorerAction(
                    action,
                    candidate.node_id,
                    f"observable candidate {candidate.file_path}:{candidate.symbol or ''}",
                )
            )
        return tuple(ranking), tuple(actions)


class GuidedNaturalDiagnosisEngine:
    """Run registered H10-C7 retrieval variants without any Gold channel."""

    def __init__(
        self,
        *,
        dense_encoders: Sequence[DenseCodeEncoder] = (),
        cross_encoder: CrossEncoder | None = None,
        action_budget: int = 12,
        structural_only: bool = False,
    ) -> None:
        self.dense_encoders = tuple(dense_encoders)
        self.cross_encoder = cross_encoder
        self.structural_only = structural_only
        self.bm25 = BM25Retriever()
        self.graph_ranker = RepoGraphRanker()
        self.reranker = StructuralReranker()
        self.exact_symbols = ExactSymbolExtractor()
        self.reservoir = CandidateReservoir()
        self.r9_config = R9CompressionConfig()
        self.r9_identifiers = StrictIdentifierExtractor()
        self.r9_compressor = R9CandidateCompressor(self.r9_config)
        self.r10_config = R10RetrievalConfig()
        self.r10_files = FileRetriever()
        self.r10_pool = R10SymbolPoolBuilder()
        self.r10_reranker = R10SourceAwareReranker()
        self.contracts = HierarchicalContractInferenceEngine()
        self.calibrator = EvidenceCalibrator()
        self.evidence_planner = ActiveEvidenceRequestPlanner()
        self.r10_evidence_planner = R10TargetedProbePlanner()
        self.router = IncidentRouter()
        self.explorer = BoundedRepositoryExplorer(action_budget)
        self._cache_graph_id: int | None = None
        self._cache_documents: tuple[SymbolDocument, ...] = ()
        self._cache_graph_ranking: tuple[RankedSymbol, ...] = ()
        self._cache_bm25_query = ""
        self._cache_bm25_ranking: tuple[RankedSymbol, ...] = ()
        self._cache_runtime_signature: tuple[str, ...] = ()
        self._cache_structural_query = ""
        self._cache_structural_ranking: tuple[RankedSymbol, ...] = ()

    def _documents(
        self,
        graph: RepositoryGraph,
        runtime_events: Sequence[RuntimeEvent] = (),
    ) -> tuple[SymbolDocument, ...]:
        graph_id = id(graph)
        runtime_signature = tuple(item.event_id for item in runtime_events)
        if (
            self._cache_graph_id != graph_id
            or self._cache_runtime_signature != runtime_signature
        ):
            self._cache_graph_id = graph_id
            self._cache_runtime_signature = runtime_signature
            self._cache_documents = documents_from_graph(
                graph,
                runtime_events,
            )
            self._cache_graph_ranking = self.graph_ranker.rank(
                graph,
                self._cache_documents,
            )
            self._cache_bm25_query = ""
            self._cache_bm25_ranking = ()
            self._cache_structural_query = ""
            self._cache_structural_ranking = ()
        return self._cache_documents

    def _graph_ranking(
        self,
        graph: RepositoryGraph,
        documents: tuple[SymbolDocument, ...],
    ) -> tuple[RankedSymbol, ...]:
        return self._cache_graph_ranking

    def _bm25_ranking(
        self,
        graph: RepositoryGraph,
        query: IncidentQuery,
        documents: tuple[SymbolDocument, ...],
    ) -> tuple[RankedSymbol, ...]:
        if self._cache_bm25_query != query.text:
            self._cache_bm25_query = query.text
            self._cache_bm25_ranking = self.bm25.rank(
                query.text,
                documents,
            )
        return self._cache_bm25_ranking

    @staticmethod
    def _runtime_ranking(
        documents: tuple[SymbolDocument, ...],
    ) -> tuple[RankedSymbol, ...]:
        values = []
        for item in documents:
            if not item.executed:
                continue
            score = 1.0
            sources = ["executed_slice"]
            if item.traceback_distance == 0.0:
                score += 6.0
                sources.append("traceback")
            if item.dynamic_call_distance is not None:
                score += 4.0 / (1.0 + item.dynamic_call_distance)
                sources.append("dynamic_call_distance")
            if item.directed_caller_distance is not None:
                score += 3.0 / (1.0 + item.directed_caller_distance)
                sources.append("directed_caller_distance")
            if item.directed_callee_distance is not None:
                score += 1.5 / (1.0 + item.directed_callee_distance)
                sources.append("directed_callee_distance")
            score += 0.35 * min(math.log1p(item.execution_frequency), 5.0)
            score += 0.75 * item.last_touch_proximity
            score += 0.75 * item.failing_test_frequency
            score += min(len(item.obligations), 3)
            values.append(
                RankedSymbol(
                    item.node_id,
                    item.file_path,
                    item.symbol,
                    score,
                    tuple(sources),
                    item.line_count,
                    item.obligations,
                )
            )
        return tuple(
            sorted(
                values,
                key=lambda item: (
                    -item.score,
                    item.file_path,
                    item.symbol or "",
                ),
            )[:100]
        )

    @staticmethod
    def _runtime_ranking_v2(
        documents: tuple[SymbolDocument, ...],
        *,
        limit: int = 300,
    ) -> tuple[RankedSymbol, ...]:
        values = []
        for item in documents:
            if not item.executed:
                continue
            score = 0.25
            sources = ["runtime_v2"]
            if item.traceback_distance == 0.0:
                score += 5.0
                sources.append("traceback")
            if item.dynamic_call_distance is not None:
                score += 3.0 / (1.0 + item.dynamic_call_distance)
                sources.append("dynamic_call_distance")
            if item.directed_caller_distance is not None:
                score += 2.5 / (1.0 + item.directed_caller_distance)
                sources.append("directed_caller_distance")
            if item.directed_callee_distance is not None:
                score += 1.0 / (1.0 + item.directed_callee_distance)
                sources.append("directed_callee_distance")
            score += 2.0 * item.last_touch_proximity
            score += 1.25 * item.failing_test_frequency
            score -= 0.05 * math.log1p(item.execution_frequency)
            values.append(
                RankedSymbol(
                    item.node_id,
                    item.file_path,
                    item.symbol,
                    score,
                    tuple(sources),
                    item.line_count,
                    item.obligations,
                    (
                        f"r9_execution_frequency:{item.execution_frequency}",
                        f"r9_last_touch:{item.last_touch_proximity:.6f}",
                    ),
                )
            )
        return tuple(
            sorted(
                values,
                key=lambda item: (-item.score, item.file_path, item.symbol or ""),
            )[:limit]
        )

    def diagnose(
        self,
        graph: RepositoryGraph,
        query: IncidentQuery,
        variant: str,
        runtime_events: Sequence[RuntimeEvent] = (),
    ) -> GuidedDiagnosis:
        if variant not in METHODS:
            raise ValueError(f"unsupported H10-C7 variant: {variant}")
        aliases = {
            "B_GREEDY": "R0",
            "B_REPOGRAPH": "R1",
            "B_AGENTLESS_LOC": "R7",
            "O_ROUTE": "R8",
        }
        if variant in aliases:
            return replace(
                self.diagnose(
                    graph,
                    query,
                    aliases[variant],
                    runtime_events,
                ),
                variant=variant,
            )
        documents = (
            documents_from_graph(
                graph,
                runtime_events,
                source_kinds=R9_SOURCE_KINDS,
            )
            if variant in {*R9_VARIANTS, *R10_VARIANTS}
            else self._documents(graph, runtime_events)
        )
        route = self.router.route(query, graph)
        if not documents:
            return GuidedDiagnosis(
                variant,
                "INSUFFICIENT_EVIDENCE",
                (),
                (),
                route,
                unavailable_reason="repository_graph_has_no_source_symbols",
            )
        if variant == "B_TRACE":
            ranking = tuple(
                RankedSymbol(
                    item.node_id,
                    item.file_path,
                    item.symbol,
                    1.0,
                    ("traceback",),
                    item.line_count,
                    item.obligations,
                )
                for item in documents
                if item.traceback_distance == 0.0
            )
            unavailable = None
        elif variant == "B_BM25":
            ranking = self._bm25_ranking(graph, query, documents)[:20]
            unavailable = None
        elif variant == "B_DENSE":
            if not self.dense_encoders:
                ranking, unavailable = (), "no_registered_dense_encoder"
            else:
                ranking = DenseRetriever(self.dense_encoders[0]).rank(
                    query.text,
                    documents,
                    limit=20,
                )
                unavailable = None
        elif variant == "B_RRF":
            if not self.dense_encoders:
                ranking, unavailable = (), "no_registered_dense_encoder"
            else:
                ranking = reciprocal_rank_fusion(
                    (
                        self.bm25.rank(query.text, documents),
                        DenseRetriever(self.dense_encoders[0]).rank(
                            query.text,
                            documents,
                        ),
                    ),
                    limit=20,
                )
                unavailable = None
        elif variant in R10_VARIANTS:
            ranking, unavailable = self._r10_ranking(
                query,
                documents,
                runtime_events,
                variant,
            )
        else:
            ranking, unavailable = self._ranking(
                graph,
                query,
                documents,
                variant,
            )
        if unavailable:
            return GuidedDiagnosis(
                variant,
                "VARIANT_UNAVAILABLE",
                (),
                (),
                route,
                unavailable_reason=unavailable,
            )
        trajectory: tuple[ExplorerAction, ...] = ()
        if variant == "R7":
            ranking, trajectory = self.explorer.explore(query, ranking)
        if variant == "R8":
            ranking = self._route_ranking(
                route,
                graph,
                query,
                documents,
                ranking,
            )
        active_status = "NOT_REQUESTED"
        active_details: tuple[tuple[str, str], ...] = ()
        active_contract_before = ""
        if variant == "R6":
            if ranking:
                active_contract_before = self.contracts.infer(
                    query,
                    ranking[0],
                    graph,
                )[0].family
            ranking, active_status, active_details = self._apply_replay_probe(
                graph,
                ranking,
                runtime_events,
            )
        candidates = []
        for item in ranking[:20]:
            hypotheses = self.contracts.infer(query, item, graph)
            candidates.append(
                GuidedCandidate(
                    item.node_id,
                    item.file_path,
                    item.symbol,
                    item.score,
                    hypotheses[0],
                    hypotheses,
                    item.rank_sources,
                    item.line_count,
                    item.obligations,
                    item.evidence,
                )
            )
        if variant in {"R5", "R6", "R8"}:
            candidates = self._joint_candidate_contract_order(
                candidates,
                self.contracts.infer_incident(query),
            )
        if variant == "R6" and candidates:
            active_details = (
                *active_details,
                ("contract_before", active_contract_before),
                ("contract_after", candidates[0].contract.family),
            )
        requests: tuple[EvidenceRequest, ...] = ()
        if query.failing_tests and variant in {"R6", "R8"}:
            requests = self.evidence_planner.plan(
                query.failing_tests[0],
                ranking,
            )
        elif query.failing_tests and variant == "R10D":
            requests = self.r10_evidence_planner.plan(
                query.failing_tests[0],
                ranking,
            )
        status = "INSUFFICIENT_EVIDENCE"
        if candidates:
            top = candidates[0]
            has_runtime_support = bool(
                top.obligations
                or
                {"traceback", "dynamic_call_distance"}.intersection(
                    top.rank_sources
                )
            )
            status = (
                "DIAGNOSIS_CONFIRMED"
                if has_runtime_support
                and self.calibrator.is_confirmed(top)
                else "DIAGNOSIS_CANDIDATES"
            )
        return GuidedDiagnosis(
            variant,
            status,
            tuple(candidates),
            requests[
                : (
                    self.r10_config.maximum_probes
                    if variant == "R10D"
                    else 1
                )
            ],
            route,
            trajectory,
            active_evidence_status=active_status,
            active_evidence_details=active_details,
        )

    def _ranking(
        self,
        graph: RepositoryGraph,
        query: IncidentQuery,
        documents: tuple[SymbolDocument, ...],
        variant: str,
    ) -> tuple[tuple[RankedSymbol, ...], str | None]:
        if variant in R9_VARIANTS:
            return self._r9_ranking(
                graph,
                query,
                documents,
                variant,
            )
        if variant == "R0":
            return self._legacy_ranking(graph, documents), None
        graph_ranking = self._graph_ranking(graph, documents)
        if variant == "R1":
            return self.reranker.rank(graph_ranking, documents), None
        if not self.dense_encoders and not (
            self.structural_only and variant in {"R3", "R5", "R6"}
        ):
            return (), "no_registered_dense_encoder"
        bm25 = self._bm25_ranking(graph, query, documents)
        dense = tuple(
            DenseRetriever(encoder).rank(query.text, documents)
            for encoder in self.dense_encoders
        )
        fused = reciprocal_rank_fusion((bm25, *dense))
        if variant == "R2":
            return fused[:20], None
        if self._cache_structural_query != query.text:
            runtime = self._runtime_ranking(documents)
            exact = self.exact_symbols.rank(query, documents)
            legacy = self._legacy_ranking(graph, documents)
            reservoir = self.reservoir.build(
                (exact, bm25, graph_ranking, runtime, legacy, *dense),
                weights=(
                    1.5,
                    1.0,
                    0.9,
                    1.35,
                    1.25,
                    *(1.0 for _ in dense),
                ),
            )
            self._cache_structural_query = query.text
            self._cache_structural_ranking = self.reranker.rank(
                reservoir,
                documents,
                limit=100,
                query=query,
            )
        reranked = self._cache_structural_ranking
        if variant == "R3":
            return reranked, None
        if variant == "R4":
            if self.cross_encoder is None:
                return (), "no_registered_cross_encoder"
            return self._cross_rerank(query, documents, reranked), None
        # R5-R8 retain the observable R3 pool when the optional cross-encoder
        # is unavailable; availability is recorded in the method matrix.
        if self.cross_encoder is not None:
            reranked = self._cross_rerank(query, documents, reranked)
        globally_ordered = self._global_order(reranked, graph.obligations)
        return (
            self._contract_pair_ranking(
                query,
                graph,
                globally_ordered,
            ),
            None,
        )

    def _r9_ranking(
        self,
        graph: RepositoryGraph,
        query: IncidentQuery,
        documents: tuple[SymbolDocument, ...],
        variant: str,
    ) -> tuple[tuple[RankedSymbol, ...], str | None]:
        config = self.r9_config
        channels = self._r9_channels(
            graph,
            query,
            documents,
            include_dense=variant == "R9C",
        )
        hierarchical = variant in {"R9B", "R9C"}
        if variant == "R9C":
            if not self.dense_encoders:
                return (), "no_registered_dense_encoder"
            if self.cross_encoder is None:
                return (), "no_registered_cross_encoder"
            top_40 = self.r9_compressor.rank(
                channels,
                documents,
                hierarchical=True,
                limit=config.rerank_limit,
            )
            channels["cross"] = self._cross_rerank(
                query,
                documents,
                top_40,
            )
        ranking = self.r9_compressor.rank(
            channels,
            documents,
            hierarchical=hierarchical,
            limit=config.final_limit,
        )
        return ranking, None

    def _r9_channels(
        self,
        graph: RepositoryGraph,
        query: IncidentQuery,
        documents: tuple[SymbolDocument, ...],
        *,
        include_dense: bool = False,
    ) -> dict[str, tuple[RankedSymbol, ...]]:
        config = self.r9_config
        channels: dict[str, tuple[RankedSymbol, ...]] = {
            "bm25": self.bm25.rank(
                query.text,
                documents,
                limit=config.bm25_limit,
            ),
            "repograph": self.graph_ranker.rank(
                graph,
                documents,
                limit=config.graph_limit,
            ),
            "runtime": self._runtime_ranking_v2(
                documents,
                limit=config.runtime_limit,
            ),
            "strict_identifier": self.r9_identifiers.rank(
                query,
                documents,
                limit=config.strict_identifier_limit,
            ),
            "legacy": self._legacy_ranking(
                graph,
                documents,
                limit=config.legacy_limit,
            ),
        }
        if include_dense and self.dense_encoders:
            dense_rankings = tuple(
                DenseRetriever(encoder).rank(
                    query.text,
                    documents,
                    limit=config.dense_limit,
                )
                for encoder in self.dense_encoders
            )
            channels["dense"] = reciprocal_rank_fusion(
                dense_rankings,
                limit=config.dense_limit,
            )
        return channels

    def _r10_ranking(
        self,
        query: IncidentQuery,
        documents: tuple[SymbolDocument, ...],
        runtime_events: Sequence[RuntimeEvent],
        variant: str,
    ) -> tuple[tuple[RankedSymbol, ...], str | None]:
        config = self.r10_config
        files = self.r10_files.rank(
            query,
            documents,
            runtime_events,
            limit=config.file_limit,
        )
        pool = self.r10_pool.build(
            query,
            files,
            documents,
            symbols_per_file=config.symbols_per_file,
            pool_limit=config.pool_limit,
        )
        if variant == "R10A":
            ranking = self.r10_reranker.rerank(
                pool,
                documents,
                runtime_events,
            )
            return ranking[: config.final_limit], None
        if self.cross_encoder is None:
            return (), "no_registered_source_aware_cross_encoder"
        selected = pool[: config.semantic_rerank_limit]
        by_id = {document.node_id: document for document in documents}
        semantic_documents = [by_id[item.node_id] for item in selected]
        semantic_scores = self.cross_encoder.score(
            self._r10_semantic_query(query),
            semantic_documents,
        )
        ranking = self.r10_reranker.rerank(
            selected,
            documents,
            runtime_events if variant in {"R10C", "R10D"} else (),
            semantic_scores=semantic_scores,
        )
        return ranking[: config.final_limit], None

    @staticmethod
    def _r10_semantic_query(query: IncidentQuery) -> str:
        normalized = IncidentNormalizer().normalize(query)
        return (
            f"[INCIDENT]\n{normalized.issue}\n"
            f"[FAILING TEST]\n{normalized.failing_tests}\n"
            f"[TRACEBACK]\n{normalized.traceback}\n"
            f"[ASSERTION]\n{normalized.assertion}\n"
            f"[EXCEPTION]\n{normalized.exception}"
        )

    def _contract_pair_ranking(
        self,
        query: IncidentQuery,
        graph: RepositoryGraph,
        ranking: Sequence[RankedSymbol],
    ) -> tuple[RankedSymbol, ...]:
        incident = {
            prediction.family: prediction.confidence
            for prediction in self.contracts.infer_incident(query)
        }
        values = []
        for item in ranking:
            hypotheses = self.contracts.infer(query, item, graph)
            compatibility = 0.0
            evidence_bonus = 0.0
            for hypothesis in hypotheses:
                pair_score = (
                    hypothesis.confidence
                    * incident.get(hypothesis.family, 0.0)
                )
                compatibility = max(compatibility, pair_score)
                compatibility_evidence = [
                    evidence
                    for evidence in hypothesis.evidence
                    if evidence.startswith("candidate_compatibility:")
                ]
                if pair_score and compatibility_evidence:
                    match_count = max(
                        evidence.count("+") + 1
                        for evidence in compatibility_evidence
                    )
                    evidence_bonus = max(
                        evidence_bonus,
                        hypothesis.confidence
                        * (1.0 + 0.25 * (match_count - 1)),
                    )
            values.append(
                RankedSymbol(
                    item.node_id,
                    item.file_path,
                    item.symbol,
                    item.score
                    + 0.45 * compatibility
                    + 0.85 * evidence_bonus,
                    (*item.rank_sources, "candidate_contract_pair"),
                    item.line_count,
                    item.obligations,
                    (
                        *item.evidence,
                        f"contract_pair_compatibility:{compatibility:.6f}",
                        f"candidate_contract_evidence:{evidence_bonus:.6f}",
                    ),
                )
            )
        return tuple(
            sorted(
                values,
                key=lambda item: (-item.score, item.file_path, item.symbol or ""),
            )
        )

    @staticmethod
    def _joint_candidate_contract_order(
        candidates: Sequence[GuidedCandidate],
        incident_hypotheses: Sequence[ContractPrediction],
    ) -> list[GuidedCandidate]:
        incident_families = {
            prediction.family: prediction.confidence
            for prediction in incident_hypotheses
        }

        def score(candidate: GuidedCandidate) -> tuple[float, float, str]:
            compatibility = max(
                (
                    prediction.confidence
                    * incident_families.get(prediction.family, 0.0)
                    for prediction in candidate.contract_hypotheses
                ),
                default=0.0,
            )
            runtime = float(
                bool(
                    {"traceback", "dynamic_call_distance", "executed_slice"}
                    .intersection(candidate.rank_sources)
                )
            )
            return (
                candidate.score + 0.30 * compatibility + 0.05 * runtime,
                compatibility,
                candidate.node_id,
            )

        return sorted(candidates, key=score, reverse=True)

    @staticmethod
    def _apply_replay_probe(
        graph: RepositoryGraph,
        ranking: Sequence[RankedSymbol],
        runtime_events: Sequence[RuntimeEvent],
    ) -> tuple[
        tuple[RankedSymbol, ...],
        str,
        tuple[tuple[str, str], ...],
    ]:
        if not runtime_events or len(ranking) < 2:
            return (
                tuple(ranking),
                "ACTIVE_EVIDENCE_UNAVAILABLE",
                (("reason", "no_withheld_runtime_observation"),),
            )
        nodes = tuple(graph.nodes)
        by_key = {
            (node.file_path, node.symbol): node.node_id
            for node in nodes
            if node.file_path
        }
        candidate_ids = {item.node_id for item in ranking[:20]}
        start = max(0, int(len(runtime_events) * 0.75))
        observed: list[str] = []
        selected_event = None
        for event in reversed(runtime_events[start:]):
            if event.kind != "traceback_frame":
                continue
            identifiers = (
                by_key.get((event.source_file, event.source_symbol)),
                by_key.get((event.target_file, event.target_symbol)),
            )
            matched = [
                node_id
                for node_id in identifiers
                if node_id in candidate_ids
                and node_id != ranking[0].node_id
            ]
            if matched:
                observed = matched
                selected_event = event
                break
        if selected_event is None:
            return (
                tuple(ranking),
                "ACTIVE_EVIDENCE_UNAVAILABLE",
                (("reason", "probe_did_not_distinguish_candidates"),),
            )
        reranked = apply_probe_observation(
            ranking,
            observed,
            confidence=0.65,
        )
        before = next(
            index
            for index, item in enumerate(ranking, start=1)
            if item.node_id == observed[0]
        )
        after = next(
            index
            for index, item in enumerate(reranked, start=1)
            if item.node_id == observed[0]
        )
        before_entropy = GuidedNaturalDiagnosisEngine._ranking_entropy(ranking)
        after_entropy = GuidedNaturalDiagnosisEngine._ranking_entropy(reranked)
        return (
            reranked,
            "ACTIVE_EVIDENCE_APPLIED",
            (
                ("event_id", selected_event.event_id),
                ("event_kind", selected_event.kind),
                ("observed_node", observed[0]),
                ("rank_before", str(before)),
                ("rank_after", str(after)),
                ("entropy_before", f"{before_entropy:.9f}"),
                ("entropy_after", f"{after_entropy:.9f}"),
            ),
        )

    @staticmethod
    def _ranking_entropy(ranking: Sequence[RankedSymbol]) -> float:
        scores = [item.score for item in ranking[:3]]
        if not scores:
            return 0.0
        maximum = max(scores)
        exponentials = [math.exp(value - maximum) for value in scores]
        total = sum(exponentials)
        probabilities = [value / total for value in exponentials]
        return -sum(
            probability * math.log(probability)
            for probability in probabilities
            if probability
        )

    @staticmethod
    def _legacy_ranking(
        graph: RepositoryGraph,
        documents: Sequence[SymbolDocument],
        *,
        limit: int = 20,
    ) -> tuple[RankedSymbol, ...]:
        by_id = {item.node_id: item for item in documents}
        values = []
        for item in EvidenceGroundedCandidateRetriever(
            max_candidates=limit
        ).retrieve(graph):
            document = by_id.get(item.node_id)
            if document is None or item.file_path is None:
                continue
            values.append(
                RankedSymbol(
                    item.node_id,
                    item.file_path,
                    item.symbol,
                    item.retrieval_score,
                    ("h10_c5c_retriever",),
                    document.line_count,
                    item.covered_obligations,
                )
            )
        return tuple(values)

    def _cross_rerank(
        self,
        query: IncidentQuery,
        documents: Sequence[SymbolDocument],
        ranking: Sequence[RankedSymbol],
    ) -> tuple[RankedSymbol, ...]:
        assert self.cross_encoder is not None
        by_id = {item.node_id: item for item in documents}
        selected = [by_id[item.node_id] for item in ranking]
        scores = self.cross_encoder.score(query.text, selected)
        values = [
            RankedSymbol(
                item.node_id,
                item.file_path,
                item.symbol,
                float(score),
                (
                    *item.rank_sources,
                    f"cross:{self.cross_encoder.model_name}@{self.cross_encoder.revision}",
                ),
                item.line_count,
                item.obligations,
            )
            for item, score in zip(ranking, scores)
        ]
        return tuple(
            sorted(
                values,
                key=lambda item: (-item.score, item.file_path, item.symbol or ""),
            )
        )

    @staticmethod
    def _global_order(
        ranking: Sequence[RankedSymbol],
        obligations: Sequence[str],
    ) -> tuple[RankedSymbol, ...]:
        remaining = set(obligations)
        selected: list[RankedSymbol] = []
        available = list(ranking)
        while remaining and available:
            winner = max(
                available,
                key=lambda item: (
                    len(remaining & set(item.obligations)),
                    item.score,
                    -item.line_count,
                    item.node_id,
                ),
            )
            if not remaining.intersection(winner.obligations):
                break
            selected.append(winner)
            remaining.difference_update(winner.obligations)
            available.remove(winner)
        return (*selected, *available)

    def _route_ranking(
        self,
        route: RoutingDecision,
        graph: RepositoryGraph,
        query: IncidentQuery,
        documents: Sequence[SymbolDocument],
        ranking: tuple[RankedSymbol, ...],
    ) -> tuple[RankedSymbol, ...]:
        if route.route == "EXECUTED_SLICE_GRAPH":
            graph_ranking = self._graph_ranking(graph, documents)
            return self.reranker.rank(
                reciprocal_rank_fusion((ranking, graph_ranking)),
                documents,
            )
        if route.route in {
            "CONFIGURATION_MATCHER",
            "DEPENDENCY_RESOLVER",
            "SERIALIZATION_PATH",
        }:
            lexical = self._bm25_ranking(graph, query, tuple(documents))
            return reciprocal_rank_fusion((lexical, ranking), limit=20)
        return ranking
