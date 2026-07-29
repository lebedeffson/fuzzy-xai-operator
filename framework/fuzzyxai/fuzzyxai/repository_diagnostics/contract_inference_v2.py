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
    "DATA_TYPE": ("dtype", "typeerror", "unexpected type"),
    "DATA_SHAPE": ("dimension", "ndim", "shape", "size mismatch"),
    "DEPENDENCY_VERSION": ("dependency", "incompatible version", "requires", "version"),
    "PIPELINE_CONFIGURATION": ("config", "configuration", "option", "setting"),
    "ARTIFACT_PROVENANCE": ("artifact", "metadata", "path", "provenance"),
    "ARTIFACT_CHECKSUM": ("checksum", "digest", "hash mismatch"),
    "SERIALIZATION_FORMAT": ("decode", "deserialize", "pickle", "serialize"),
    "MODEL_LOADING": ("checkpoint", "load model", "state_dict"),
    "MODEL_EXPLAINER_VERSION": ("explainer version", "model version"),
    "RUNTIME_API": ("api", "attributeerror", "deprecated", "signature"),
}


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
        text = " ".join(
            (
                query.text,
                candidate.file_path,
                candidate.symbol or "",
                *(
                    item.detail
                    for item in graph.evidence
                    if item.kind
                    in {
                        "runtime_assertion",
                        "runtime_exception",
                        "runtime_traceback_frame",
                        "traceback",
                    }
                ),
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
