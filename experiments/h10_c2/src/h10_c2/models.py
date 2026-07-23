from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass(frozen=True)
class Transaction:
    transaction_id: str
    operation: str
    target_kind: str
    target_id: str
    field: str | None
    before: Any
    after: Any
    inverse: dict[str, Any]
    repair_cost: float
    repairable: bool = True


@dataclass(frozen=True)
class GoldRecord:
    case_id: str
    gold_status: str
    optimal_cost: float | None
    optimal_cuts: tuple[tuple[str, ...], ...]
    covered_obligations: tuple[str, ...]
    repairable: bool
    allowed_repairs: tuple[dict[str, Any], ...]
    oracle_trace_sha256: str


@dataclass(frozen=True)
class MethodResult:
    case_id: str
    pipeline: str
    method: str
    predicted_cut: tuple[str, ...]
    predicted_cost: float
    repair_actions: tuple[dict[str, Any], ...] = ()
    optimality_claimed: bool = False
    status: str = "ok"
    runtime_ms: float = 0.0
    memory_bytes: int = 0


@dataclass(frozen=True)
class Case:
    case_id: str
    pipeline: str
    modality: str
    split: str
    case_type: str
    clean_route: dict[str, Any]
    observed_route: dict[str, Any]
    public_obligations: tuple[dict[str, Any], ...]
    repair_costs: dict[str, float]
    case_hash: str
    transactions: tuple[Transaction, ...] = field(default_factory=tuple, repr=False)

    def public_dict(self) -> dict[str, Any]:
        value = asdict(self)
        value.pop("transactions", None)
        value.pop("clean_route", None)
        value.pop("public_obligations", None)
        return value

    def method_view(self) -> "Case":
        return Case(
            case_id=self.case_id,
            pipeline=self.pipeline,
            modality=self.modality,
            split=self.split,
            case_type=self.case_type,
            clean_route={},
            observed_route=self.observed_route,
            public_obligations=(),
            repair_costs=self.repair_costs,
            case_hash=self.case_hash,
            transactions=(),
        )
