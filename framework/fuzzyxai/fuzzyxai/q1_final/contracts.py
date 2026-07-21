"""Typed contracts for the final evidence and release boundary."""

from __future__ import annotations

import re
from dataclasses import asdict, dataclass
from enum import Enum
from typing import Mapping, Sequence


COMMIT_RE = re.compile(r"^[0-9a-f]{40}$")


class ClaimLevel(str, Enum):
    CONTROLLED = "controlled"
    REAL = "real"
    HUMAN = "human"
    DOMAIN = "domain"


class ClaimStatusV2(str, Enum):
    SUPPORTED = "supported"
    NOT_SUPPORTED = "not_supported"
    INCONCLUSIVE = "inconclusive"
    EXTERNAL_GATE = "external_gate"
    REMOVED = "removed"


@dataclass(frozen=True)
class FinalRunIdentity:
    schema_version: str
    branch: str
    base_commit: str
    final_commit: str
    ci_run_ids: tuple[str, ...]
    profile: str
    real_benchmark_status: str
    external_gate_status: Mapping[str, str]
    stable_release_allowed: bool
    created_at: str
    python: str
    platform: str
    threads: int

    def __post_init__(self) -> None:
        if self.schema_version != "2.0":
            raise ValueError("FinalRunIdentity requires schema version 2.0")
        if not COMMIT_RE.fullmatch(self.base_commit) or not COMMIT_RE.fullmatch(self.final_commit):
            raise ValueError("base_commit and final_commit must be complete Git commit hashes")
        if self.profile != "full_q1_final":
            raise ValueError("final identity must use the full_q1_final profile")
        if self.real_benchmark_status not in {"pass", "fail", "not_run"}:
            raise ValueError("invalid real benchmark status")
        if self.threads != 1:
            raise ValueError("Q1 final evidence must record single-thread orchestration")
        closed_external = {"supported", "not_supported", "inconclusive"}
        if self.stable_release_allowed and any(value not in closed_external for value in self.external_gate_status.values()):
            raise ValueError("stable release cannot be enabled while an external gate is open")

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True)
class ClaimRecordV2:
    claim_id: str
    level: ClaimLevel
    status: ClaimStatusV2
    claim: str
    allowed_wording_ru: str
    allowed_wording_en: str
    forbidden_wording: tuple[str, ...]
    datasets: tuple[str, ...]
    models: tuple[str, ...]
    n_objects: int
    n_seeds: int
    metrics: Mapping[str, object]
    confidence_intervals: Mapping[str, object]
    evidence: tuple[str, ...]
    limitations: tuple[str, ...]
    final_commit: str

    def __post_init__(self) -> None:
        if not self.claim_id or not self.claim or not self.evidence:
            raise ValueError("claim id, text and evidence are required")
        if not COMMIT_RE.fullmatch(self.final_commit):
            raise ValueError("claim final_commit must be a complete Git hash")
        if self.n_objects < 0 or self.n_seeds < 0:
            raise ValueError("claim sample sizes cannot be negative")
        if self.status is ClaimStatusV2.SUPPORTED and not (self.allowed_wording_ru and self.allowed_wording_en):
            raise ValueError("supported claims require bilingual allowed wording")

    def public_wording(self, language: str = "ru") -> str:
        if self.status is not ClaimStatusV2.SUPPORTED:
            raise RuntimeError(f"claim {self.claim_id} is not supported")
        return self.allowed_wording_ru if language == "ru" else self.allowed_wording_en

    def to_dict(self) -> dict[str, object]:
        payload = asdict(self)
        payload["level"] = self.level.value
        payload["status"] = self.status.value
        return payload


@dataclass(frozen=True)
class StructuralDiagnosticResult:
    indicator_id: str
    fault_types: tuple[str, ...]
    source_locations: tuple[str, ...]
    detection_latency_ms: float
    evidence_refs: tuple[str, ...]

    def __post_init__(self) -> None:
        if not self.indicator_id or not self.evidence_refs:
            raise ValueError("structural result requires an id and evidence")
        if self.detection_latency_ms < 0:
            raise ValueError("detection latency cannot be negative")


@dataclass(frozen=True)
class PredictionAssociationResult:
    m0_auprc: float
    m1_auprc: float
    incremental_auprc: float
    evaluation_partition: str
    predictive_claim_allowed: bool
    evidence_refs: tuple[str, ...]

    def __post_init__(self) -> None:
        if self.evaluation_partition != "test":
            raise ValueError("prediction association must be evaluated on test")
        if self.predictive_claim_allowed and self.incremental_auprc <= 0:
            raise ValueError("non-positive incremental AUPRC cannot support a predictive claim")


@dataclass(frozen=True)
class ExternalGateRecord:
    gate_id: str
    status: str
    required_count: int
    observed_count: int
    ethics_status: str
    raw_records: tuple[str, ...]
    signed_records: tuple[str, ...]
    scorer_output: str | None
    claim_removed_if_not_supported: bool

    def __post_init__(self) -> None:
        statuses = {"open", "supported", "not_supported", "inconclusive", "failed"}
        if self.status not in statuses:
            raise ValueError(f"invalid external gate status: {self.status}")
        if self.observed_count < 0 or self.required_count <= 0:
            raise ValueError("invalid external study counts")
        closed = self.status in {"supported", "not_supported", "inconclusive"}
        if closed:
            if self.ethics_status not in {"approved", "exempt"}:
                raise ValueError("a closed external gate requires ethics approval or exemption")
            if self.observed_count < self.required_count:
                raise ValueError("a closed external gate lacks required independent records")
            if not self.raw_records or not self.signed_records or not self.scorer_output:
                raise ValueError("a closed external gate requires raw, signed and scored evidence")
        if self.status in {"not_supported", "inconclusive"} and not self.claim_removed_if_not_supported:
            raise ValueError("a non-positive gate requires removal or limitation of its claim")


def ensure_same_commit(records: Sequence[Mapping[str, object]], final_commit: str) -> None:
    if not COMMIT_RE.fullmatch(final_commit):
        raise ValueError("invalid final commit")
    mismatches = [record.get("final_commit") for record in records if record.get("final_commit") != final_commit]
    if mismatches:
        raise ValueError(f"artifact identity mismatch: {mismatches}")
