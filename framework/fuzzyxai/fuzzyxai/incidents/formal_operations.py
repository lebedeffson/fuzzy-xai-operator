from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum


class FormalOperation(StrEnum):
    OPEN_LOG = "OPEN_LOG"
    OPEN_MANIFEST = "OPEN_MANIFEST"
    READ_VERSION = "READ_VERSION"
    COMPARE_VERSION = "COMPARE_VERSION"
    READ_SCHEMA = "READ_SCHEMA"
    COMPARE_SCHEMA = "COMPARE_SCHEMA"
    VERIFY_HASH = "VERIFY_HASH"
    TRACE_DEPENDENCY = "TRACE_DEPENDENCY"
    RUN_TEST = "RUN_TEST"
    FORM_HYPOTHESIS = "FORM_HYPOTHESIS"
    SELECT_COMPONENT = "SELECT_COMPONENT"
    EXECUTE_REPAIR = "EXECUTE_REPAIR"
    VERIFY_POSTCONDITION = "VERIFY_POSTCONDITION"
    RUN_RECERTIFICATION = "RUN_RECERTIFICATION"


@dataclass(frozen=True)
class OperationEvent:
    operation_id: FormalOperation
    precondition: str
    input_evidence: tuple[str, ...]
    output_evidence: tuple[str, ...]
    measured_runtime_ms: float
    dependency_cost: float


def operation_cost(event: OperationEvent, mode: str = "uniform") -> float:
    if mode == "uniform":
        return 1.0
    if mode == "runtime":
        return event.measured_runtime_ms
    if mode == "dependency":
        return 1.0 + event.dependency_cost
    raise ValueError(f"unsupported operation cost mode: {mode}")
