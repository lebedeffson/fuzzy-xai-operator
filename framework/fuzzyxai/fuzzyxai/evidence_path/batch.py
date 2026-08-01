from __future__ import annotations

import hashlib
import json
import threading
import time
from dataclasses import asdict, dataclass
from typing import Literal

import numpy as np

from .digest import merkle_root, tensor_digest

AuditMode = Literal["online", "full"]


@dataclass(frozen=True)
class StaticManifest:
    artifact_id: str
    version: str
    model_digest: str
    explainer_digest: str
    preprocessing_digest: str
    feature_schema_digest: str
    route_digest: str
    contract_registry_digest: str

    @property
    def cache_key(self) -> str:
        fingerprint = merkle_root(
            (name, str(value))
            for name, value in asdict(self).items()
            if name not in {"artifact_id", "version"}
        )
        return f"{self.artifact_id}:{self.version}:{fingerprint}"


@dataclass(frozen=True)
class SampleAuditRecord:
    sample_id: str
    prediction_digest: str
    explanation_digest: str
    dynamic_root: str
    status: str


@dataclass(frozen=True)
class BatchAuditReport:
    mode: AuditMode
    static_root: str
    sample_records: tuple[SampleAuditRecord, ...]
    merkle_root: str
    status: str
    timings_ms: dict[str, float]
    serialized: bytes | None


class StaticArtifactCache:
    def __init__(self) -> None:
        self._values: dict[str, str] = {}
        self._lock = threading.Lock()

    def resolve(self, manifest: StaticManifest) -> str:
        key = manifest.cache_key
        cached = self._values.get(key)
        if cached is not None:
            return cached
        root = merkle_root(
            (
                ("model", manifest.model_digest),
                ("explainer", manifest.explainer_digest),
                ("preprocessing", manifest.preprocessing_digest),
                ("feature_schema", manifest.feature_schema_digest),
                ("route", manifest.route_digest),
                ("contract_registry", manifest.contract_registry_digest),
            )
        )
        with self._lock:
            return self._values.setdefault(key, root)

    def __len__(self) -> int:
        return len(self._values)


def _milliseconds(start: int, end: int) -> float:
    return (end - start) / 1_000_000


def audit_batch(
    predictions: object,
    explanations: object,
    shared_manifest: StaticManifest,
    sample_ids: tuple[str, ...],
    *,
    cache: StaticArtifactCache,
    mode: AuditMode = "online",
) -> BatchAuditReport:
    if mode not in {"online", "full"}:
        raise ValueError("mode must be online or full")
    started = time.perf_counter_ns()
    prediction_array = np.ascontiguousarray(np.asarray(predictions))
    explanation_array = np.ascontiguousarray(np.asarray(explanations))
    normalized = time.perf_counter_ns()
    if prediction_array.ndim == 0 or explanation_array.ndim == 0:
        raise ValueError("batch evidence requires a leading sample dimension")
    if prediction_array.shape[0] != explanation_array.shape[0]:
        raise ValueError("prediction and explanation batch sizes differ")
    if prediction_array.shape[0] != len(sample_ids):
        raise ValueError("sample_ids do not match the batch size")
    if len(set(sample_ids)) != len(sample_ids):
        raise ValueError("sample_ids must be unique")

    prediction_digests = tuple(tensor_digest(prediction_array[index]) for index in range(len(sample_ids)))
    prediction_done = time.perf_counter_ns()
    explanation_digests = tuple(tensor_digest(explanation_array[index]) for index in range(len(sample_ids)))
    explanation_done = time.perf_counter_ns()

    contract_checks = (
        prediction_array.dtype.hasobject is False
        and explanation_array.dtype.hasobject is False
        and all(sample_ids)
    )
    contracts_done = time.perf_counter_ns()
    static_root = cache.resolve(shared_manifest)
    records = tuple(
        SampleAuditRecord(
            sample_id,
            prediction_digest,
            explanation_digest,
            merkle_root(
                (
                    ("sample_identity", hashlib.sha256(sample_id.encode()).hexdigest()),
                    ("prediction", prediction_digest),
                    ("explanation", explanation_digest),
                )
            ),
            "supported" if contract_checks else "insufficient_evidence",
        )
        for sample_id, prediction_digest, explanation_digest in zip(
            sample_ids,
            prediction_digests,
            explanation_digests,
            strict=True,
        )
    )
    route_done = time.perf_counter_ns()
    # Online mode validates a pre-registered route; no dynamic cut is needed.
    cut_done = time.perf_counter_ns()
    root = merkle_root(
        (
            ("static", static_root),
            *(
                (f"sample:{record.sample_id}", record.dynamic_root)
                for record in records
            ),
        )
    )
    proof_done = time.perf_counter_ns()
    serialized = None
    if mode == "full":
        # Archive serialization is deliberately outside the online path.
        serialized = (
            '{"merkle_root":"'
            + root
            + '","mode":"full","samples":['
            + ",".join(
                (
                    '{"dynamic_root":"'
                    + record.dynamic_root
                    + '","explanation_digest":"'
                    + record.explanation_digest
                    + '","prediction_digest":"'
                    + record.prediction_digest
                    + '","sample_id":'
                    + json.dumps(record.sample_id, ensure_ascii=True)
                    + ',"status":"'
                    + record.status
                    + '"}'
                )
                for record in records
            )
            + '],"static_root":"'
            + static_root
            + '"}'
        ).encode()
    serialization_done = time.perf_counter_ns()
    timings = {
        "normalize_ms": _milliseconds(started, normalized),
        "prediction_digest_ms": _milliseconds(normalized, prediction_done),
        "explanation_digest_ms": _milliseconds(prediction_done, explanation_done),
        "contract_check_ms": _milliseconds(explanation_done, contracts_done),
        "route_build_ms": _milliseconds(contracts_done, route_done),
        "cut_search_ms": _milliseconds(route_done, cut_done),
        "proof_trace_ms": _milliseconds(cut_done, proof_done),
        "serialization_ms": _milliseconds(proof_done, serialization_done),
    }
    return BatchAuditReport(
        mode,
        static_root,
        records,
        root,
        "supported" if contract_checks else "insufficient_evidence",
        timings,
        serialized,
    )
