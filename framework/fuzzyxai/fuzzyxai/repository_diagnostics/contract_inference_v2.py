from __future__ import annotations

import math
import re
from collections import defaultdict
from dataclasses import dataclass

from .graph import RepositoryGraph
from .guided_retrieval import (
    IncidentNormalizer,
    IncidentQuery,
    RankedSymbol,
    tokenize,
)

CONTRACT_HIERARCHY = {
    "DATA": ("DATA_SCHEMA", "DATA_TYPE", "DATA_SHAPE"),
    "DEPENDENCY": ("DEPENDENCY_VERSION",),
    "CONFIGURATION": (
        "PIPELINE_CONFIGURATION",
        "OUTPUT_FORMAT",
        "CONTROL_FLOW",
        "FILTERING_POLICY",
        "PROTOCOL_RESPONSE",
        "PATH_VALIDATION",
        "SELECTION_POLICY",
        "DEFAULT_VALUE_POLICY",
        "STATE_TRANSITION",
        "API_BEHAVIOR",
    ),
    "ARTIFACT": ("ARTIFACT_PROVENANCE", "ARTIFACT_CHECKSUM"),
    "SERIALIZATION": ("SERIALIZATION_FORMAT",),
    "MODEL": ("MODEL_LOADING", "MODEL_EXPLAINER_VERSION"),
    "EXPLAINER": ("MODEL_EXPLAINER_VERSION",),
    "RUNTIME": ("RUNTIME_API",),
    "UNKNOWN": ("UNKNOWN_CONTRACT",),
}

RULES = {
    "DATA_SCHEMA": ("column", "field", "schema"),
    "DATA_TYPE": ("cannot safely cast", "dtype", "unexpected type"),
    "DATA_SHAPE": ("dimension", "ndim", "shape", "size mismatch"),
    "DEPENDENCY_VERSION": ("dependency", "incompatible version", "requires", "version"),
    "PIPELINE_CONFIGURATION": (
        "config",
        "configuration",
        "exit status",
        "header",
        "option",
        "output_path",
        "setting",
    ),
    "OUTPUT_FORMAT": (
        "carriage return",
        "line ending",
        "newline",
        "output format",
        "whitespace",
    ),
    "CONTROL_FLOW": (
        "index out of range",
        "pop index",
        "empty sequence",
        "iteration",
    ),
    "FILTERING_POLICY": (
        "exclude",
        "filter",
        "omitting",
        "password",
        "remove duplicates",
    ),
    "PROTOCOL_RESPONSE": (
        "header",
        "host",
        "response",
        "status code",
    ),
    "PATH_VALIDATION": (
        "absolute path",
        "path traversal",
        "validate path",
    ),
    "SELECTION_POLICY": (
        "selector",
        "sorted",
        "priority",
        "order",
    ),
    "DEFAULT_VALUE_POLICY": (
        "default",
        "missing",
        "none",
    ),
    "STATE_TRANSITION": (
        "initialize",
        "state transition",
        "lifecycle",
    ),
    "API_BEHAVIOR": (
        "attributeerror",
        "expected behavior",
        "public api",
        "return value",
    ),
    "ARTIFACT_PROVENANCE": ("artifact", "metadata", "provenance"),
    "ARTIFACT_CHECKSUM": ("checksum", "digest", "hash mismatch"),
    "SERIALIZATION_FORMAT": ("decode", "deserialize", "pickle", "serialize"),
    "MODEL_LOADING": ("checkpoint", "load model", "state_dict"),
    "MODEL_EXPLAINER_VERSION": ("explainer version", "model version"),
    "RUNTIME_API": ("deprecated api", "missing attribute", "signature mismatch"),
}

EVALUATION_FAMILY = {
    "DATA_SCHEMA": "DATA_CONTRACT",
    "DATA_TYPE": "DATA_CONTRACT",
    "DATA_SHAPE": "DATA_CONTRACT",
    "PIPELINE_CONFIGURATION": "CONFIGURATION",
    "OUTPUT_FORMAT": "CONFIGURATION",
    "CONTROL_FLOW": "CONFIGURATION",
    "FILTERING_POLICY": "CONFIGURATION",
    "PROTOCOL_RESPONSE": "CONFIGURATION",
    "PATH_VALIDATION": "CONFIGURATION",
    "SELECTION_POLICY": "CONFIGURATION",
    "DEFAULT_VALUE_POLICY": "CONFIGURATION",
    "STATE_TRANSITION": "CONFIGURATION",
    "API_BEHAVIOR": "CONFIGURATION",
    "SERIALIZATION_FORMAT": "SERIALIZATION",
}


def evaluation_contract_family(family: str) -> str:
    """Map fine-grained predictions to the published development ontology."""
    return EVALUATION_FAMILY.get(family, family)


@dataclass(frozen=True)
class ContractPrediction:
    family: str
    coarse_family: str
    confidence: float
    evidence: tuple[str, ...]


class HierarchicalContractInferenceEngine:
    def infer_incident(
        self,
        query: IncidentQuery,
    ) -> tuple[ContractPrediction, ...]:
        normalized_incident = IncidentNormalizer().normalize(query)
        normalized = " ".join(
            tokenize(
                " ".join(
                    (
                        normalized_incident.assertion,
                        normalized_incident.exception,
                        normalized_incident.traceback[-6000:],
                        normalized_incident.issue,
                    )
                )
            )
        )
        scores: dict[str, float] = defaultdict(float)
        reasons: dict[str, set[str]] = defaultdict(set)
        self._apply_rules(normalized, scores, reasons)
        self._apply_direct_observations(normalized, scores, reasons)
        raw_assertion = normalized_incident.assertion.lower()
        if "\\r\\n" in raw_assertion and "\\n" in raw_assertion:
            scores["OUTPUT_FORMAT"] += 8.0
            reasons["OUTPUT_FORMAT"].add(
                "direct_observation:line_ending_bytes"
            )
        if "422" in raw_assertion and any(
            term in normalized
            for term in ("form", "field", "param", "sequence", "list")
        ):
            scores["DATA_SCHEMA"] += 8.0
            reasons["DATA_SCHEMA"].add(
                "direct_observation:validation_status_422"
            )
        if any(
            term in normalized
            for term in (
                "at index diff",
                "extra items in the left set",
                "hashable",
                "test equality",
            )
        ):
            scores["DATA_SCHEMA"] += 6.0
            reasons["DATA_SCHEMA"].add(
                "direct_observation:collection_semantics"
            )
        if "type" in normalized and any(
            term in normalized
            for term in ("import as names", "expected node", "parse")
        ):
            scores["DATA_SCHEMA"] += 6.0
            reasons["DATA_SCHEMA"].add(
                "direct_observation:parser_node_type"
            )
        if not scores:
            return (self._unknown(),)
        return self._rank(scores, reasons)

    def infer(
        self,
        query: IncidentQuery,
        candidate: RankedSymbol,
        graph: RepositoryGraph,
    ) -> tuple[ContractPrediction, ...]:
        incident = self.infer_incident(query)
        scores: dict[str, float] = defaultdict(float)
        reasons: dict[str, set[str]] = defaultdict(set)
        incident_supported = any(
            prediction.family != "UNKNOWN_CONTRACT"
            for prediction in incident
        )
        for prediction in incident:
            if prediction.family == "UNKNOWN_CONTRACT":
                continue
            scores[prediction.family] += 5.0 * prediction.confidence
            reasons[prediction.family].update(prediction.evidence)
            reasons[prediction.family].add("incident_level_hypothesis")
        candidate_text = " ".join(
            tokenize(f"{candidate.file_path} {candidate.symbol or ''}")
        )
        if incident_supported:
            self._apply_rules(candidate_text, scores, reasons, weight=0.50)
        symbol = (candidate.symbol or "").lower()
        path = candidate.file_path.lower()
        compatibility = {
            "OUTPUT_FORMAT": ("format", "render", "write"),
            "CONTROL_FLOW": ("iter", "next", "pop", "sequence"),
            "FILTERING_POLICY": ("filter", "exclude", "field", "clone"),
            "PROTOCOL_RESPONSE": (
                "header",
                "request",
                "response",
                "router",
                "websocket",
            ),
            "PATH_VALIDATION": ("path", "file", "url"),
            "SELECTION_POLICY": ("select", "sort", "priority", "nearest"),
            "DEFAULT_VALUE_POLICY": ("default", "missing", "optional"),
            "STATE_TRANSITION": ("initialize", "state", "loop"),
            "API_BEHAVIOR": ("api", "call", "command", "encode"),
            "DATA_SCHEMA": ("field", "schema", "frame", "args"),
            "DATA_TYPE": ("dtype", "cast", "integer", "array"),
            "SERIALIZATION_FORMAT": ("serialize", "decode", "escape"),
            "ARTIFACT_PROVENANCE": ("artifact", "url", "path", "session"),
        }
        if incident_supported:
            for family, terms in compatibility.items():
                matches = [
                    term for term in terms if term in symbol or term in path
                ]
                if matches:
                    scores[family] += 0.6 + 0.2 * len(matches)
                    reasons[family].add(
                        f"candidate_compatibility:{'+'.join(matches)}"
                    )
        node = graph.node(candidate.node_id)
        if node is not None:
            if node.kind == "dependency":
                scores["DEPENDENCY_VERSION"] += 4.0
                reasons["DEPENDENCY_VERSION"].add("node_kind:dependency")
            if node.kind == "configuration_key":
                scores["PIPELINE_CONFIGURATION"] += 4.0
                reasons["PIPELINE_CONFIGURATION"].add(
                    "node_kind:configuration_key"
                )
        if not scores:
            return (self._unknown(),)
        return self._rank(scores, reasons)

    @staticmethod
    def _apply_rules(
        normalized: str,
        scores: dict[str, float],
        reasons: dict[str, set[str]],
        *,
        weight: float = 1.0,
    ) -> None:
        for family, patterns in RULES.items():
            for pattern in patterns:
                tokens = " ".join(tokenize(pattern))
                count = len(re.findall(rf"\b{re.escape(tokens)}\b", normalized))
                if count:
                    scores[family] += weight * (1.0 + math.log1p(count))
                    reasons[family].add(f"observed:{pattern}")

    @staticmethod
    def _apply_direct_observations(
        normalized: str,
        scores: dict[str, float],
        reasons: dict[str, set[str]],
    ) -> None:
        observations = {
            "DATA_SCHEMA": (
                ("status code", "422"),
                ("column", "different"),
            ),
            "DATA_TYPE": (
                ("float64", "int64"),
                ("boolean value", "ambiguous"),
                ("dtype", "different"),
            ),
            "FILTERING_POLICY": (
                ("password", "differing items"),
                ("extra items",),
            ),
            "OUTPUT_FORMAT": (
                ("not found", "newline"),
                ("whitespace",),
            ),
            "PROTOCOL_RESPONSE": (
                ("status code",),
                ("response", "assert"),
                ("header",),
            ),
            "CONTROL_FLOW": (
                ("indexerror",),
                ("pop index",),
                ("list index",),
            ),
            "ARTIFACT_PROVENANCE": (
                ("urljoin",),
                ("rtmp",),
                ("path traversal",),
            ),
            "SERIALIZATION_FORMAT": (
                ("unicode escape",),
                ("reader", "writer"),
                ("decode",),
            ),
            "API_BEHAVIOR": (
                ("attributeerror",),
                ("nameerror",),
            ),
        }
        for family, variants in observations.items():
            for terms in variants:
                if all(" ".join(tokenize(term)) in normalized for term in terms):
                    scores[family] += 5.0
                    reasons[family].add(
                        f"direct_observation:{'+'.join(terms)}"
                    )

    @staticmethod
    def _unknown() -> ContractPrediction:
        return ContractPrediction(
            "UNKNOWN_CONTRACT",
            "UNKNOWN",
            0.0,
            ("no_registered_contract_supported",),
        )

    @staticmethod
    def _rank(
        scores: dict[str, float],
        reasons: dict[str, set[str]],
    ) -> tuple[ContractPrediction, ...]:
        maximum = max(scores.values())
        ranked = []
        for family, score in sorted(
            scores.items(),
            key=lambda item: (-item[1], item[0]),
        ):
            coarse = next(
                name
                for name, children in CONTRACT_HIERARCHY.items()
                if family in children
            )
            ranked.append(
                ContractPrediction(
                    family,
                    coarse,
                    score / max(maximum, 1e-12),
                    tuple(sorted(reasons[family])),
                )
            )
        return tuple(ranked)
