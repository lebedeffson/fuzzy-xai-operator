"""Typed contracts for action-conditioned route certification."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Mapping


class ContractOutcome(str, Enum):
    SATISFIED = "satisfied"
    UNSATISFIED = "unsatisfied"
    NOT_APPLICABLE = "not_applicable"


@dataclass(frozen=True)
class ContractRequirement:
    contract_id: str
    expected: str
    source_path: str
    severity: float = 1.0
    repair_cost: float = 1.0
    repairable: bool = True
    blocking: bool = True

    def __post_init__(self) -> None:
        if not self.contract_id or not self.source_path:
            raise ValueError("contract identity and source path are required")
        if not 0.0 <= self.severity <= 1.0 or self.repair_cost < 0.0:
            raise ValueError("severity must be in [0, 1] and repair cost cannot be negative")


@dataclass(frozen=True)
class ContractCheck:
    requirement: ContractRequirement
    actual: str | None
    outcome: ContractOutcome
    reason_code: str

    @property
    def satisfied(self) -> bool:
        return self.outcome is ContractOutcome.SATISFIED


@dataclass(frozen=True)
class AuditFeatureVector:
    certificate_exists: float
    certificate_size: float
    certificate_depth: float
    certificate_coverage: float
    number_of_unsatisfied_contracts: float
    weighted_contract_severity: float
    minimal_cut_size: float
    minimal_repair_cost: float
    number_of_independent_fault_sources: float
    path_redundancy: float
    source_conflict_count: float
    provenance_completeness: float
    reference_population_distance: float
    artifact_age: float
    version_distance: float
    canonical_integrity: float
    route_entropy: float

    def as_mapping(self) -> Mapping[str, float]:
        return self.__dict__
