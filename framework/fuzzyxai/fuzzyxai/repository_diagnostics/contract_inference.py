from __future__ import annotations

import math
from dataclasses import dataclass

from .graph import RepositoryGraph
from .retrieval import RetrievedCandidate

SUPPORTED_CONTRACTS = (
    "ARTIFACT_PROVENANCE",
    "DATA_CONTRACT",
    "DEPENDENCY_VERSION",
    "MODEL_EXPLAINER_VERSION",
    "MODEL_LOADING",
    "PIPELINE_CONFIGURATION",
    "SERIALIZATION",
)


@dataclass(frozen=True)
class ContractCandidate:
    family: str
    confidence: float
    score: float
    evidence: tuple[str, ...]


@dataclass(frozen=True)
class ContractInference:
    contract: str
    score: float
    confidence: float
    evidence_reasons: tuple[str, ...]
    supported: bool


class EvidenceGroundedContractInferer:
    """Rank contract families independently from file/symbol retrieval."""

    def infer(
        self,
        graph: RepositoryGraph,
        candidate: RetrievedCandidate,
    ) -> ContractInference:
        winner = self.infer_candidates(graph, candidate)[0]
        return ContractInference(
            winner.family,
            winner.score,
            winner.confidence,
            winner.evidence,
            winner.family != "UNREGISTERED_CONTRACT",
        )

    def infer_candidates(
        self,
        graph: RepositoryGraph,
        candidate: RetrievedCandidate,
    ) -> tuple[ContractCandidate, ...]:
        node = graph.node(candidate.node_id)
        if node is None:
            return self._unregistered("candidate_missing_from_graph")

        scores = {name: 0.0 for name in SUPPORTED_CONTRACTS}
        reasons = {name: set() for name in SUPPORTED_CONTRACTS}
        relations = {edge.relation for edge in graph.edges if edge.source == node.node_id or edge.target == node.node_id}
        runtime_text = " ".join(
            item.detail.lower()
            for item in graph.evidence
            if item.kind
            in {
                "failing_test",
                "runtime_assertion",
                "runtime_exception",
                "runtime_traceback_frame",
                "traceback",
            }
        )
        tokens = {str(value).lower() for value in node.attributes.get("semantic_tokens", ())}
        symbol = (node.symbol or "").lower()

        self._add(
            scores,
            reasons,
            "DEPENDENCY_VERSION",
            8.0,
            "dependency_node",
            node.kind == "dependency",
        )
        self._add(
            scores,
            reasons,
            "PIPELINE_CONFIGURATION",
            7.0,
            "configuration_key",
            node.kind == "configuration_key",
        )
        self._add(
            scores,
            reasons,
            "SERIALIZATION",
            6.0,
            "serialized_artifact",
            node.kind == "serialized_artifact",
        )
        self._add(
            scores,
            reasons,
            "MODEL_LOADING",
            7.0,
            "model_checkpoint",
            node.kind == "model_checkpoint",
        )
        self._add(
            scores,
            reasons,
            "MODEL_EXPLAINER_VERSION",
            8.0,
            "model_explainer_version_mismatch",
            _version_mismatch(node.attributes),
        )
        self._add(
            scores,
            reasons,
            "SERIALIZATION",
            5.0,
            "serialization_symbol",
            bool(
                tokens
                & {
                    "decode",
                    "deserialize",
                    "dump",
                    "dumps",
                    "encode",
                    "load",
                    "loads",
                    "pickle",
                    "serialize",
                }
            ),
        )
        self._add(
            scores,
            reasons,
            "DATA_CONTRACT",
            5.0,
            "schema_type_shape_tokens",
            bool(
                tokens
                & {
                    "columns",
                    "dtype",
                    "fields",
                    "ndim",
                    "schema",
                    "shape",
                    "validate",
                }
            ),
        )
        self._add(
            scores,
            reasons,
            "ARTIFACT_PROVENANCE",
            4.0,
            "artifact_path_tokens",
            bool(
                tokens
                & {
                    "artifact",
                    "cache",
                    "checksum",
                    "digest",
                    "metadata",
                    "path",
                }
            ),
        )
        self._add(
            scores,
            reasons,
            "PIPELINE_CONFIGURATION",
            3.0,
            "configuration_tokens",
            bool(tokens & {"config", "configuration", "options", "settings"}),
        )
        self._add(
            scores,
            reasons,
            "SERIALIZATION",
            4.0,
            "serialization_relation",
            "serializes" in relations,
        )
        self._add(
            scores,
            reasons,
            "ARTIFACT_PROVENANCE",
            3.0,
            "runtime_io_relation",
            bool(relations & {"loads", "reads", "writes"}),
        )
        self._add(
            scores,
            reasons,
            "DEPENDENCY_VERSION",
            3.0,
            "dependency_relation",
            "depends_on" in relations,
        )
        self._add(
            scores,
            reasons,
            "PIPELINE_CONFIGURATION",
            3.0,
            "configuration_relation",
            "configured_by" in relations,
        )
        self._add(
            scores,
            reasons,
            "DATA_CONTRACT",
            2.0,
            "runtime_type_or_assertion",
            _data_runtime(runtime_text) and bool(tokens & {"dtype", "fields", "shape", "validate"}),
        )
        self._add(
            scores,
            reasons,
            "SERIALIZATION",
            2.0,
            "runtime_serialization",
            any(token in runtime_text for token in ("decode", "pickle", "serializ")) and any(token in symbol for token in ("decode", "load", "serial")),
        )

        ranked = tuple(
            ContractCandidate(
                name,
                1.0 - math.exp(-score / 8.0),
                score,
                tuple(sorted(reasons[name])),
            )
            for name, score in sorted(
                scores.items(),
                key=lambda item: (-item[1], item[0]),
            )
            if score >= 3.0
        )
        return ranked or self._unregistered("contract_family_not_supported_by_evidence")

    @staticmethod
    def _unregistered(reason: str) -> tuple[ContractCandidate, ...]:
        return (
            ContractCandidate(
                "UNREGISTERED_CONTRACT",
                0.0,
                0.0,
                (reason,),
            ),
        )

    @staticmethod
    def _add(
        scores: dict[str, float],
        reasons: dict[str, set[str]],
        contract: str,
        value: float,
        reason: str,
        condition: bool,
    ) -> None:
        if condition:
            scores[contract] += value
            reasons[contract].add(reason)


ContractInferenceEngine = EvidenceGroundedContractInferer


def _version_mismatch(attributes: dict[str, object]) -> bool:
    model = str(attributes.get("model_version", ""))
    explainer = str(attributes.get("explainer_version", ""))
    return bool(model and explainer and model != explainer)


def _data_runtime(runtime_text: str) -> bool:
    return any(
        token in runtime_text
        for token in (
            "assert",
            "dtype",
            "field",
            "shape",
            "typeerror",
            "valueerror",
        )
    )
