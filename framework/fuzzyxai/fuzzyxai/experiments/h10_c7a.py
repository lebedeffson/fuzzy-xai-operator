from __future__ import annotations

import json
import statistics
import time
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

from fuzzyxai.experiments.h10_c7 import (
    FORBIDDEN_OBSERVABLE_KEYS,
    GoldAtom,
    GoldLocalization,
    _graph,
    _reject_gold,
)
from fuzzyxai.repository_diagnostics.contract_inference_v2 import (
    evaluation_contract_family,
)
from fuzzyxai.repository_diagnostics.graph import RepositoryGraph
from fuzzyxai.repository_diagnostics.guided_diagnosis import (
    GuidedCandidate,
    GuidedDiagnosis,
    GuidedNaturalDiagnosisEngine,
)
from fuzzyxai.repository_diagnostics.guided_retrieval import (
    DenseRetriever,
    IncidentQuery,
    RankedSymbol,
    SymbolDocument,
    reciprocal_rank_fusion,
)
from fuzzyxai.repository_diagnostics.retrieval import (
    EvidenceGroundedCandidateRetriever,
)
from fuzzyxai.repository_diagnostics.runtime_events import (
    RuntimeEvent,
    load_runtime_events,
)

BUDGETS = (5, 10, 20, 40, 80, 160)
METHODS = (
    "B_TRACE",
    "B_BM25",
    "B_DENSE",
    "B_RRF",
    "B_GREEDY",
    "B_REPOGRAPH",
    "R0",
    "R5",
)
STRUCTURAL_METHODS = tuple(
    method for method in METHODS if method not in {"B_DENSE", "B_RRF"}
)


@dataclass(frozen=True)
class BudgetCase:
    incident_id: str
    repository: str
    query: IncidentQuery
    graph: RepositoryGraph
    runtime_events: tuple[RuntimeEvent, ...]
    repository_symbol_count: int
    repository_source_lines: int


@dataclass(frozen=True)
class BudgetRanking:
    method: str
    ranking: tuple[RankedSymbol, ...]
    frozen_diagnosis: GuidedDiagnosis
    runtime_ms: float
    unavailable_reason: str | None = None


def _read_jsonl(path: Path) -> tuple[dict[str, object], ...]:
    return tuple(
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    )


def _resolve(base: Path, value: object) -> Path:
    path = Path(str(value))
    return path if path.is_absolute() else (base / path).resolve()


def _repository_source_lines(graph: RepositoryGraph) -> int:
    maxima: dict[str, int] = {}
    for node in graph.nodes:
        if not node.file_path or node.kind not in {"class", "function", "method"}:
            continue
        end = int(node.attributes.get("end_lineno", 0) or 0)
        maxima[node.file_path] = max(maxima.get(node.file_path, 0), end)
    return sum(maxima.values())


def load_budget_inputs(
    manifest_path: Path,
    gold_path: Path,
    *,
    minimum_incidents: int = 1,
    minimum_repositories: int = 1,
) -> tuple[tuple[BudgetCase, ...], dict[str, GoldLocalization]]:
    observable = _read_jsonl(manifest_path)
    for index, value in enumerate(observable):
        _reject_gold(value, f"$[{index}]")
        leaked = FORBIDDEN_OBSERVABLE_KEYS.intersection(value)
        if leaked:
            raise ValueError(f"observable Gold leakage: {sorted(leaked)}")
    cases = []
    for value in observable:
        if value.get("split") != "development":
            raise ValueError("H10-C7A budget scoring accepts development only")
        query = value["query"]
        if not isinstance(query, dict):
            raise TypeError("query must be a mapping")
        graph_value = value.get("graph")
        if graph_value is None:
            graph_value = json.loads(
                _resolve(manifest_path.parent, value["graph_path"]).read_text(
                    encoding="utf-8"
                )
            )
        if not isinstance(graph_value, dict):
            raise TypeError("graph must be a mapping")
        graph = _graph(graph_value)
        runtime_path = _resolve(
            manifest_path.parent,
            value["runtime_events_path"],
        )
        cases.append(
            BudgetCase(
                incident_id=str(value["incident_id"]),
                repository=str(value["repository"]),
                query=IncidentQuery(
                    str(value["incident_id"]),
                    str(query.get("issue", "")),
                    tuple(str(item) for item in query.get("failing_tests", ())),
                    str(query.get("traceback", "")),
                    str(query.get("assertion", "")),
                ),
                graph=graph,
                runtime_events=load_runtime_events(runtime_path),
                repository_symbol_count=int(value["repository_symbol_count"]),
                repository_source_lines=int(
                    value.get("repository_source_lines", 0)
                    or _repository_source_lines(graph)
                ),
            )
        )
    gold = {}
    for value in _read_jsonl(gold_path):
        identifier = str(value["incident_id"])
        gold[identifier] = GoldLocalization(
            identifier,
            tuple(
                GoldAtom(
                    str(atom["file_path"]),
                    str(atom["symbol"]) if atom.get("symbol") is not None else None,
                    str(atom["contract"]),
                )
                for atom in value["atoms"]
            ),
        )
    identifiers = [case.incident_id for case in cases]
    if len(identifiers) != len(set(identifiers)):
        raise ValueError("H10-C7A incident IDs must be unique")
    if set(identifiers) != set(gold):
        raise ValueError("H10-C7A observable and Gold incident sets differ")
    repositories = {case.repository for case in cases}
    if len(cases) < minimum_incidents:
        raise ValueError(
            f"H10-C7A requires at least {minimum_incidents} incidents"
        )
    if len(repositories) < minimum_repositories:
        raise ValueError(
            f"H10-C7A requires at least {minimum_repositories} repositories"
        )
    return tuple(cases), gold


def _as_ranked(candidate: GuidedCandidate) -> RankedSymbol:
    return RankedSymbol(
        candidate.node_id,
        candidate.file_path,
        candidate.symbol,
        candidate.score,
        candidate.rank_sources,
        candidate.line_count,
        candidate.obligations,
        candidate.evidence,
    )


def _append_tail(
    frozen: Sequence[RankedSymbol],
    extended: Sequence[RankedSymbol],
    *,
    limit: int,
) -> tuple[RankedSymbol, ...]:
    values = list(frozen)
    seen = {item.node_id for item in values}
    for item in extended:
        if item.node_id in seen:
            continue
        values.append(item)
        seen.add(item.node_id)
        if len(values) >= limit:
            break
    return tuple(values)


def _legacy_ranking(
    graph: RepositoryGraph,
    documents: Sequence[SymbolDocument],
    *,
    limit: int,
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


class FrozenBudgetRankingEngine:
    """Expose larger budgets while preserving every frozen top-20 prefix."""

    def __init__(
        self,
        diagnosis_engine: GuidedNaturalDiagnosisEngine,
        *,
        maximum_budget: int = max(BUDGETS),
    ) -> None:
        self.engine = diagnosis_engine
        self.maximum_budget = maximum_budget

    def rank(
        self,
        case: BudgetCase,
        method: str,
    ) -> BudgetRanking:
        if method not in METHODS:
            raise ValueError(f"unsupported H10-C7A method: {method}")
        started = time.perf_counter_ns()
        diagnosis = self.engine.diagnose(
            case.graph,
            case.query,
            method,
            case.runtime_events,
        )
        if diagnosis.status == "VARIANT_UNAVAILABLE":
            return BudgetRanking(
                method,
                (),
                diagnosis,
                (time.perf_counter_ns() - started) / 1_000_000,
                diagnosis.unavailable_reason,
            )
        frozen = tuple(_as_ranked(item) for item in diagnosis.candidates)
        documents = self.engine._documents(
            case.graph,
            case.runtime_events,
        )
        extended = self._extended(case, method, documents)
        ranking = _append_tail(
            frozen,
            extended,
            limit=self.maximum_budget,
        )
        elapsed = (time.perf_counter_ns() - started) / 1_000_000
        return BudgetRanking(method, ranking, diagnosis, elapsed)

    def _extended(
        self,
        case: BudgetCase,
        method: str,
        documents: tuple[SymbolDocument, ...],
    ) -> tuple[RankedSymbol, ...]:
        graph = case.graph
        query = case.query
        limit = self.maximum_budget
        if method == "B_TRACE":
            return tuple(
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
        if method == "B_BM25":
            return self.engine.bm25.rank(query.text, documents, limit=limit)
        if method == "B_DENSE":
            if not self.engine.dense_encoders:
                return ()
            return DenseRetriever(self.engine.dense_encoders[0]).rank(
                query.text,
                documents,
                limit=limit,
            )
        if method == "B_RRF":
            if not self.engine.dense_encoders:
                return ()
            dense = DenseRetriever(self.engine.dense_encoders[0]).rank(
                query.text,
                documents,
                limit=limit,
            )
            return reciprocal_rank_fusion(
                (
                    self.engine.bm25.rank(query.text, documents, limit=limit),
                    dense,
                ),
                limit=limit,
            )
        if method in {"B_GREEDY", "R0"}:
            return _legacy_ranking(graph, documents, limit=limit)
        graph_ranking = self.engine.graph_ranker.rank(
            graph,
            documents,
            limit=limit,
        )
        if method == "B_REPOGRAPH":
            return self.engine.reranker.rank(
                graph_ranking,
                documents,
                limit=limit,
                query=query,
            )
        if method != "R5":
            raise AssertionError(method)
        bm25 = self.engine.bm25.rank(query.text, documents, limit=limit)
        runtime = self.engine._runtime_ranking(documents)
        exact = self.engine.exact_symbols.rank(query, documents)
        legacy = _legacy_ranking(graph, documents, limit=limit)
        dense = tuple(
            DenseRetriever(encoder).rank(query.text, documents, limit=limit)
            for encoder in self.engine.dense_encoders
        )
        reservoir = self.engine.reservoir.build(
            (exact, bm25, graph_ranking, runtime, legacy, *dense),
            limit=300,
            weights=(
                1.5,
                1.0,
                0.9,
                1.35,
                1.25,
                *(1.0 for _ in dense),
            ),
        )
        reranked = self.engine.reranker.rank(
            reservoir,
            documents,
            limit=300,
            query=query,
        )
        globally_ordered = self.engine._global_order(
            reranked,
            graph.obligations,
        )
        return self.engine._contract_pair_ranking(
            query,
            graph,
            globally_ordered,
        )


def _gold_rank(
    ranking: Sequence[RankedSymbol],
    gold: GoldLocalization,
) -> int | None:
    for rank, candidate in enumerate(ranking, start=1):
        if any(
            candidate.file_path == atom.file_path
            and candidate.symbol == atom.symbol
            for atom in gold.atoms
        ):
            return rank
    return None


def budget_rows(
    cases: Sequence[BudgetCase],
    gold: dict[str, GoldLocalization],
    *,
    engine: FrozenBudgetRankingEngine,
    methods: Sequence[str] = METHODS,
) -> tuple[list[dict[str, object]], dict[str, dict[str, object]]]:
    rows: list[dict[str, object]] = []
    frozen: dict[str, dict[str, object]] = {}
    for case in cases:
        for method in methods:
            result = engine.rank(case, method)
            diagnosis = result.frozen_diagnosis
            predicted_contract = (
                evaluation_contract_family(
                    diagnosis.candidates[0].contract.family
                )
                if diagnosis.candidates
                else "UNKNOWN_CONTRACT"
            )
            gold_contracts = sorted(
                {atom.contract for atom in gold[case.incident_id].atoms}
            )
            frozen[f"{case.incident_id}:{method}"] = {
                "top_10": [item.node_id for item in result.ranking[:10]],
                "top_20": [item.node_id for item in result.ranking[:20]],
                "predicted_contract": predicted_contract,
                "status": diagnosis.status,
            }
            rank = _gold_rank(result.ranking, gold[case.incident_id])
            for budget in BUDGETS:
                selected = result.ranking[:budget]
                candidate_count = len(selected)
                context_lines = sum(item.line_count for item in selected)
                hit = bool(rank and rank <= budget)
                rows.append(
                    {
                        "incident_id": case.incident_id,
                        "repository": case.repository,
                        "method": method,
                        "budget": budget,
                        "available": not bool(result.unavailable_reason),
                        "unavailable_reason": result.unavailable_reason or "",
                        "rank": rank or 0,
                        "recall": float(hit),
                        "reciprocal_rank": (
                            1.0 / rank if rank and rank <= budget else 0.0
                        ),
                        "candidate_count": candidate_count,
                        "context_lines": context_lines,
                        "repository_symbols": case.repository_symbol_count,
                        "repository_source_lines": case.repository_source_lines,
                        "search_space_reduction": 1.0
                        - candidate_count / max(case.repository_symbol_count, 1),
                        "context_reduction": 1.0
                        - context_lines / max(case.repository_source_lines, 1),
                        "coverage": float(bool(selected)),
                        "contract_correct": float(
                            predicted_contract in gold_contracts
                        ),
                        "symbol_hit_at_3": float(bool(rank and rank <= 3)),
                        "joint_hit_at_3": float(
                            bool(
                                rank
                                and rank <= 3
                                and predicted_contract in gold_contracts
                            )
                        ),
                        "predicted_contract": predicted_contract,
                        "gold_contracts": json.dumps(gold_contracts),
                        "false_localization": float(
                            diagnosis.status == "DIAGNOSIS_CONFIRMED"
                            and not (
                                rank
                                and rank <= 3
                                and predicted_contract in gold_contracts
                            )
                        ),
                        "runtime_ms": result.runtime_ms,
                    }
                )
    return rows, frozen


def summarize_budget_rows(
    rows: Sequence[dict[str, object]],
) -> list[dict[str, object]]:
    summary = []
    keys = sorted({(str(row["method"]), int(row["budget"])) for row in rows})
    for method, budget in keys:
        selected = [
            row
            for row in rows
            if row["method"] == method and int(row["budget"]) == budget
        ]
        available = [row for row in selected if row["available"]]
        labels = sorted(
            {
                *(
                    label
                    for row in available
                    for label in json.loads(str(row["gold_contracts"]))
                ),
                *(str(row["predicted_contract"]) for row in available),
            }
        )
        contract_f1 = []
        for label in labels:
            true_positive = sum(
                label in json.loads(str(row["gold_contracts"]))
                and row["predicted_contract"] == label
                for row in available
            )
            false_positive = sum(
                label not in json.loads(str(row["gold_contracts"]))
                and row["predicted_contract"] == label
                for row in available
            )
            false_negative = sum(
                label in json.loads(str(row["gold_contracts"]))
                and row["predicted_contract"] != label
                for row in available
            )
            denominator = 2 * true_positive + false_positive + false_negative
            contract_f1.append(
                2 * true_positive / denominator if denominator else 0.0
            )
        summary.append(
            {
                "method": method,
                "budget": budget,
                "incident_count": len(selected),
                "available_incident_count": len(available),
                "repository_count": len(
                    {str(row["repository"]) for row in selected}
                ),
                "recall": statistics.fmean(
                    float(row["recall"]) for row in available
                )
                if available
                else 0.0,
                "mrr": statistics.fmean(
                    float(row["reciprocal_rank"]) for row in available
                )
                if available
                else 0.0,
                "coverage": statistics.fmean(
                    float(row["coverage"]) for row in selected
                )
                if selected
                else 0.0,
                "contract_macro_f1": (
                    statistics.fmean(contract_f1) if contract_f1 else 0.0
                ),
                "joint_hit_at_3": statistics.fmean(
                    float(row["joint_hit_at_3"]) for row in available
                )
                if available
                else 0.0,
                "mean_candidate_count": statistics.fmean(
                    float(row["candidate_count"]) for row in available
                )
                if available
                else 0.0,
                "mean_context_lines": statistics.fmean(
                    float(row["context_lines"]) for row in available
                )
                if available
                else 0.0,
                "mean_search_space_reduction": statistics.fmean(
                    float(row["search_space_reduction"]) for row in available
                )
                if available
                else 0.0,
                "mean_context_reduction": statistics.fmean(
                    float(row["context_reduction"]) for row in available
                )
                if available
                else 0.0,
                "false_localization": statistics.fmean(
                    float(row["false_localization"]) for row in selected
                )
                if selected
                else 0.0,
                "median_runtime_ms": statistics.median(
                    float(row["runtime_ms"]) for row in available
                )
                if available
                else 0.0,
            }
        )
    return summary


def select_budget_locks(
    summary: Sequence[dict[str, object]],
    *,
    minimum_recall: float = 0.80,
) -> dict[str, object]:
    selected: dict[str, dict[str, object]] = {}
    unavailable = []
    for method in METHODS:
        values = [
            row
            for row in summary
            if row["method"] == method
            and int(row["available_incident_count"])
            == int(row["incident_count"])
        ]
        eligible = [
            row for row in values if float(row["recall"]) >= minimum_recall
        ]
        if eligible:
            selected[method] = min(eligible, key=lambda row: int(row["budget"]))
        elif not values:
            unavailable.append(method)
    baselines = [
        method
        for method in selected
        if method not in {"R5"} and method.startswith(("B_", "R0"))
    ]
    baseline = (
        min(
            baselines,
            key=lambda method: (
                int(selected[method]["budget"]),
                -float(selected[method]["recall"]),
                float(selected[method]["mean_search_space_reduction"]),
                method,
            ),
        )
        if baselines
        else None
    )
    return {
        "minimum_recall": minimum_recall,
        "method_budgets": {
            method: {
                "k_star": int(row["budget"]),
                "recall": float(row["recall"]),
                "mean_search_space_reduction": float(
                    row["mean_search_space_reduction"]
                ),
            }
            for method, row in sorted(selected.items())
        },
        "selected_baseline": baseline,
        "dominance_endpoint": baseline is None,
        "unavailable_optional_methods": sorted(unavailable),
    }
