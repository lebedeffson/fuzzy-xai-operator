from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, replace
from typing import Protocol

from .active_evidence import ActiveEvidenceRequestPlanner, EvidenceRequest
from .contract_inference_v2 import (
    ContractPrediction,
    HierarchicalContractInferenceEngine,
)
from .graph import RepositoryGraph
from .guided_retrieval import (
    BM25Retriever,
    DenseCodeEncoder,
    DenseRetriever,
    IncidentQuery,
    RankedSymbol,
    RepoGraphRanker,
    StructuralReranker,
    SymbolDocument,
    documents_from_graph,
    reciprocal_rank_fusion,
)
from .incident_router import IncidentRouter, RoutingDecision
from .retrieval import EvidenceGroundedCandidateRetriever

VARIANTS = tuple(f"R{index}" for index in range(9))
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
METHODS = (*VARIANTS, *BASELINES)


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
    rank_sources: tuple[str, ...]
    line_count: int
    obligations: tuple[str, ...]


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
    ) -> None:
        self.dense_encoders = tuple(dense_encoders)
        self.cross_encoder = cross_encoder
        self.bm25 = BM25Retriever()
        self.graph_ranker = RepoGraphRanker()
        self.reranker = StructuralReranker()
        self.contracts = HierarchicalContractInferenceEngine()
        self.evidence_planner = ActiveEvidenceRequestPlanner()
        self.router = IncidentRouter()
        self.explorer = BoundedRepositoryExplorer(action_budget)

    def diagnose(
        self,
        graph: RepositoryGraph,
        query: IncidentQuery,
        variant: str,
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
                self.diagnose(graph, query, aliases[variant]),
                variant=variant,
            )
        documents = documents_from_graph(graph)
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
            ranking = self.bm25.rank(query.text, documents, limit=20)
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
        candidates = tuple(
            GuidedCandidate(
                item.node_id,
                item.file_path,
                item.symbol,
                item.score,
                self.contracts.infer(query, item, graph)[0],
                item.rank_sources,
                item.line_count,
                item.obligations,
            )
            for item in ranking[:20]
        )
        requests = (
            self.evidence_planner.plan(query.failing_tests[0], ranking)
            if variant in {"R6", "R8"} and query.failing_tests
            else ()
        )
        status = (
            "DIAGNOSIS_CANDIDATES"
            if candidates
            else "INSUFFICIENT_EVIDENCE"
        )
        return GuidedDiagnosis(
            variant,
            status,
            candidates,
            requests[:1],
            route,
            trajectory,
        )

    def _ranking(
        self,
        graph: RepositoryGraph,
        query: IncidentQuery,
        documents: tuple[SymbolDocument, ...],
        variant: str,
    ) -> tuple[tuple[RankedSymbol, ...], str | None]:
        if variant == "R0":
            return self._legacy_ranking(graph, documents), None
        graph_ranking = self.graph_ranker.rank(graph, documents)
        if variant == "R1":
            return self.reranker.rank(graph_ranking, documents), None
        if not self.dense_encoders:
            return (), "no_registered_dense_encoder"
        bm25 = self.bm25.rank(query.text, documents)
        dense = [
            DenseRetriever(encoder).rank(query.text, documents)
            for encoder in self.dense_encoders
        ]
        fused = reciprocal_rank_fusion((bm25, *dense))
        if variant == "R2":
            return fused[:20], None
        fused = reciprocal_rank_fusion((fused, graph_ranking))
        reranked = self.reranker.rank(fused, documents)
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
        return self._global_order(reranked, graph.obligations), None

    @staticmethod
    def _legacy_ranking(
        graph: RepositoryGraph,
        documents: Sequence[SymbolDocument],
    ) -> tuple[RankedSymbol, ...]:
        by_id = {item.node_id: item for item in documents}
        values = []
        for item in EvidenceGroundedCandidateRetriever(
            max_candidates=20
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
            graph_ranking = self.graph_ranker.rank(graph, documents)
            return self.reranker.rank(
                reciprocal_rank_fusion((ranking, graph_ranking)),
                documents,
            )
        if route.route in {
            "CONFIGURATION_MATCHER",
            "DEPENDENCY_RESOLVER",
            "SERIALIZATION_PATH",
        }:
            lexical = self.bm25.rank(query.text, documents)
            return reciprocal_rank_fusion((lexical, ranking), limit=20)
        return ranking
