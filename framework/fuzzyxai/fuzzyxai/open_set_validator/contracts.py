"""Contracts for open-set structural route validation."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Mapping


class OpenSetOutcome(str, Enum):
    KNOWN_FAULT_TYPE = "known_fault_type"
    UNKNOWN_STRUCTURAL_FAULT = "unknown_structural_fault"
    INSUFFICIENT_EVIDENCE = "insufficient_evidence"
    VALID_ROUTE = "valid_route"


@dataclass(frozen=True)
class StructuralObservation:
    observation_id: str
    features: Mapping[str, float]
    feature_regions: Mapping[str, str]
    missing_channels: tuple[str, ...] = ()
    source_is_oof: bool = True
    partition: str = "development"

    def __post_init__(self) -> None:
        if not self.observation_id:
            raise ValueError("observation_id is required")
        if not self.features:
            raise ValueError("structural features are required")
        if set(self.features) != set(self.feature_regions):
            raise ValueError("every structural feature must map to a source region")
        if self.partition == "test" and self.source_is_oof:
            raise ValueError("test observations cannot be marked as OOF development evidence")


@dataclass(frozen=True)
class OpenSetTrainingRow:
    observation: StructuralObservation
    fault_family: str

    def __post_init__(self) -> None:
        if self.observation.partition == "test" or not self.observation.source_is_oof:
            raise ValueError("open-set fitting requires OOF development observations")
        if not self.fault_family:
            raise ValueError("fault family is required")


@dataclass(frozen=True)
class OpenSetAssessment:
    outcome: OpenSetOutcome
    known_fault_type: str | None
    unknown_score: float
    known_confidence: float
    suspected_regions: tuple[str, ...]
    repair_candidate_set: tuple[str, ...]
    reason_codes: tuple[str, ...]
