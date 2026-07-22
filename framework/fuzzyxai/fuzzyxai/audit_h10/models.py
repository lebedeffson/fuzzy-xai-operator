from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class RouteObservation:
    route_id: str
    dataset_id: str
    modality: str
    object_id: str
    expected: dict[str, Any]
    observed: dict[str, Any]
    mandatory_fields: tuple[str, ...]
    dependency_paths: tuple[tuple[str, ...], ...]
    repair_costs: dict[str, float]


@dataclass(frozen=True)
class FaultPrediction:
    parent_family: str | None
    leaf_type: str | None
    leaf_confidence: float
    abstained_at_leaf: bool
    unknown: bool


@dataclass(frozen=True)
class DiagnosticCutResult:
    cut_nodes: tuple[str, ...]
    total_cost: float
    optimal: bool
    solver: str
    runtime_ms: float
    covered_invalid_paths: int


@dataclass(frozen=True)
class RepairAction:
    target: str
    action: str
    expected_effect: str
    preconditions: tuple[str, ...]
    affected_fields: tuple[str, ...] = ()
    verification: str = "rebuild_and_recertify_route"
    rollback: str = "restore_previous_artifact_snapshot"


@dataclass(frozen=True)
class AuditDiagnosis:
    route_status: str
    fault: FaultPrediction
    source_nodes: tuple[str, ...]
    diagnostic_cut: DiagnosticCutResult
    repair_set: tuple[RepairAction, ...]
    recertified: bool | None
    trace: bytes
    details: dict[str, Any] = field(default_factory=dict)
