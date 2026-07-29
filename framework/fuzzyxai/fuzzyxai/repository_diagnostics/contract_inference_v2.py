from __future__ import annotations

import math
import re
from collections import defaultdict
from dataclasses import dataclass

from .graph import RepositoryGraph
from .guided_retrieval import IncidentQuery, RankedSymbol, tokenize

CONTRACT_HIERARCHY = {
    "DATA": ("DATA_SCHEMA", "DATA_TYPE", "DATA_SHAPE"),
    "DEPENDENCY": ("DEPENDENCY_VERSION",),
    "CONFIGURATION": ("PIPELINE_CONFIGURATION",),
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
    def infer(
        self,
        query: IncidentQuery,
        candidate: RankedSymbol,
        graph: RepositoryGraph,
    ) -> tuple[ContractPrediction, ...]:
        traceback = "\n".join(
            line
            for line in query.traceback.splitlines()
            if not any(
                marker in line
                for marker in (
                    ".h10-c5c-development",
                    "_distutils_hack",
                    "runtime_launcher.py",
                )
            )
        )
        text = " ".join(
            (
                query.assertion,
                traceback[-6000:],
                *query.failing_tests,
                candidate.file_path,
                candidate.symbol or "",
            )
        ).lower()
        normalized = " ".join(tokenize(text))
        scores: dict[str, float] = defaultdict(float)
        reasons: dict[str, set[str]] = defaultdict(set)
        for family, patterns in RULES.items():
            for pattern in patterns:
                count = len(re.findall(rf"\b{re.escape(pattern)}\b", normalized))
                if count:
                    scores[family] += 1.0 + math.log1p(count)
                    reasons[family].add(f"observed:{pattern}")
        direct_observations = {
            "DATA_SCHEMA": (
                ("status code", "422"),
                ("extra items",),
                ("column", "different"),
            ),
            "DATA_TYPE": (
                ("float64", "int64"),
                ("boolean value", "ambiguous"),
            ),
            "ARTIFACT_PROVENANCE": (
                ("urljoin",),
                ("rtmp",),
                ("expected log message",),
            ),
            "SERIALIZATION_FORMAT": (
                ("unicode escape", "decode"),
                ("reader", "writer"),
            ),
        }
        for family, observations in direct_observations.items():
            for terms in observations:
                if all(term in normalized for term in terms):
                    scores[family] += 5.0
                    reasons[family].add(
                        f"direct_observation:{'+'.join(terms)}"
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
            return (
                ContractPrediction(
                    "UNKNOWN_CONTRACT",
                    "UNKNOWN",
                    0.0,
                    ("no_registered_contract_supported",),
                ),
            )
        maximum = max(scores.values())
        ranked = []
        for family, score in sorted(scores.items(), key=lambda item: (-item[1], item[0])):
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
