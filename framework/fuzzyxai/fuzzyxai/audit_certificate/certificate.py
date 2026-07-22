"""Action-conditioned audit certificate and feature extraction."""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import asdict, dataclass
from typing import Iterable

from .contracts import AuditFeatureVector, ContractCheck


@dataclass(frozen=True)
class ActionConditionedAuditCertificate:
    action: str
    checks: tuple[ContractCheck, ...]
    required_evidence: tuple[str, ...]
    uncertainty_constraints: tuple[str, ...]
    representation_constraints: tuple[str, ...]
    action_preconditions: tuple[str, ...]
    source_paths: tuple[str, ...]
    certificate_exists: bool
    schema_version: str = "1.0"

    def __post_init__(self) -> None:
        if self.action not in {"accept", "short_review", "full_review", "block"}:
            raise ValueError("unsupported action")
        if not self.checks:
            raise ValueError("a certificate requires at least one contract check")
        expected = all(check.satisfied or not check.requirement.blocking for check in self.checks)
        if self.certificate_exists != expected:
            raise ValueError("certificate status is inconsistent with blocking contract checks")

    @property
    def satisfied_contracts(self) -> tuple[str, ...]:
        return tuple(check.requirement.contract_id for check in self.checks if check.satisfied)

    @property
    def unsatisfied_contracts(self) -> tuple[str, ...]:
        return tuple(check.requirement.contract_id for check in self.checks if not check.satisfied)

    @property
    def sha256(self) -> str:
        return hashlib.sha256(_canonical_payload(self)).hexdigest()

    def features(
        self,
        *,
        minimal_cut_size: int = 0,
        minimal_repair_cost: float = 0.0,
        reference_population_distance: float = 0.0,
        artifact_age: float = 0.0,
        version_distance: float = 0.0,
        source_conflict_count: int = 0,
    ) -> AuditFeatureVector:
        total = len(self.checks)
        satisfied = len(self.satisfied_contracts)
        failed = [check for check in self.checks if not check.satisfied]
        severity = sum(check.requirement.severity for check in failed)
        sources = {check.requirement.source_path for check in failed}
        provenance = [check for check in self.checks if check.requirement.contract_id.startswith("provenance:")]
        provenance_complete = sum(check.satisfied for check in provenance) / max(1, len(provenance))
        canonical = next((check.satisfied for check in self.checks if check.requirement.contract_id == "canonical_integrity"), False)
        depth = max((_path_depth(path) for path in self.source_paths), default=0)
        counts = [sum(check.requirement.source_path == path for check in self.checks) for path in set(self.source_paths)]
        redundancy = 1.0 - len(set(self.source_paths)) / max(1, total)
        entropy = _normalized_entropy(counts)
        return AuditFeatureVector(
            certificate_exists=float(self.certificate_exists),
            certificate_size=float(total),
            certificate_depth=float(depth),
            certificate_coverage=satisfied / total,
            number_of_unsatisfied_contracts=float(len(failed)),
            weighted_contract_severity=float(severity),
            minimal_cut_size=float(minimal_cut_size),
            minimal_repair_cost=float(minimal_repair_cost),
            number_of_independent_fault_sources=float(len(sources)),
            path_redundancy=float(redundancy),
            source_conflict_count=float(source_conflict_count),
            provenance_completeness=float(provenance_complete),
            reference_population_distance=float(reference_population_distance),
            artifact_age=float(artifact_age),
            version_distance=float(version_distance),
            canonical_integrity=float(canonical),
            route_entropy=float(entropy),
        )


def _canonical_payload(certificate: ActionConditionedAuditCertificate) -> bytes:
    return json.dumps(asdict(certificate), sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode()


def _path_depth(path: str) -> int:
    return max(1, len([part for part in path.replace(".", "/").split("/") if part]))


def _normalized_entropy(counts: Iterable[int]) -> float:
    values = tuple(value for value in counts if value > 0)
    total = sum(values)
    if total <= 0 or len(values) <= 1:
        return 0.0
    entropy = -sum((value / total) * math.log(value / total) for value in values)
    return float(entropy / math.log(len(values)))
