from __future__ import annotations

from dataclasses import dataclass

from .graph import RepositoryGraph
from .guided_retrieval import IncidentQuery


@dataclass(frozen=True)
class RoutingDecision:
    route: str
    reasons: tuple[str, ...]


class IncidentRouter:
    def route(
        self,
        query: IncidentQuery,
        graph: RepositoryGraph,
    ) -> RoutingDecision:
        text = query.text.lower()
        obligations = len(graph.obligations)
        if obligations > 1:
            return RoutingDecision(
                "GLOBAL_MULTI_OBLIGATION",
                ("multiple_failing_obligations",),
            )
        if any(token in text for token in ("requirements", "version conflict", "dependency")):
            return RoutingDecision("DEPENDENCY_RESOLVER", ("dependency_signal",))
        if any(token in text for token in ("config", "setting", "option")):
            return RoutingDecision("CONFIGURATION_MATCHER", ("configuration_signal",))
        if any(token in text for token in ("pickle", "serialize", "deserialize")):
            return RoutingDecision("SERIALIZATION_PATH", ("serialization_signal",))
        if any(token in text for token in ("dtype", "field", "schema", "shape")):
            return RoutingDecision("DATAFLOW_ASSERTION", ("data_contract_signal",))
        if query.traceback.strip():
            return RoutingDecision("EXECUTED_SLICE_GRAPH", ("traceback_available",))
        return RoutingDecision(
            "HYBRID_RETRIEVAL_AGENT",
            ("weak_runtime_evidence",),
        )
