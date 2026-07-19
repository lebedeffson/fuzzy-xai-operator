"""Leakage-safe deterministic calibration for Q1 policies."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from itertools import product
from typing import Callable, Sequence

from .schemas import OperationKind, PartitionRole, Q1CalibrationConfig, SplitUseRecord


@dataclass(frozen=True)
class CalibrationTrial:
    config: Q1CalibrationConfig
    objective: float
    automatic_coverage: float
    runtime_cost: float

    def to_dict(self) -> dict[str, object]:
        return {"config": self.config.to_dict(), **{key: value for key, value in asdict(self).items() if key != "config"}}


def deterministic_grid_search(
    *,
    validation_hash: str,
    scorer: Callable[[Q1CalibrationConfig], tuple[float, float, float]],
    accept_grid: Sequence[float] = (0.70, 0.80, 0.90),
    conflict_grid: Sequence[float] = (0.10, 0.20, 0.30),
    escalation_grid: Sequence[float] = (0.20, 0.30, 0.40),
    cost_mode: str = "balanced",
) -> tuple[CalibrationTrial, tuple[CalibrationTrial, ...], SplitUseRecord]:
    costs = {
        "safety-heavy": (30.0, 7.0, 1.0, 2.0),
        "balanced": (20.0, 5.0, 1.0, 2.0),
        "review-expensive": (20.0, 5.0, 4.0, 2.0),
    }
    if cost_mode not in costs:
        raise ValueError(f"unknown cost mode: {cost_mode}")
    trials: list[CalibrationTrial] = []
    for accept, conflict, escalation in product(accept_grid, conflict_grid, escalation_grid):
        config = Q1CalibrationConfig(
            accept_threshold=accept,
            review_threshold=min(accept, 0.55),
            conflict_weight=1.0,
            missing_weight=1.0,
            shift_weight=1.0,
            instability_weight=1.0,
            reduction_loss_weight=1.0,
            critical_rupture_threshold=conflict,
            escalation_threshold=escalation,
            risk_cost_critical=costs[cost_mode][0],
            risk_cost_wrong_auto=costs[cost_mode][1],
            risk_cost_review=costs[cost_mode][2],
            risk_cost_false_block=costs[cost_mode][3],
        )
        objective, coverage, runtime = scorer(config)
        trials.append(CalibrationTrial(config, float(objective), float(coverage), float(runtime)))
    # Frozen tie-break: objective, then runtime, then more coverage, then lexical config.
    best = min(
        trials,
        key=lambda item: (
            item.objective,
            item.runtime_cost,
            -item.automatic_coverage,
            tuple(sorted(item.config.to_dict().items())),
        ),
    )
    record = SplitUseRecord(
        operation_id="q1-grid-calibration",
        operation=OperationKind.CALIBRATE,
        partitions=(PartitionRole.VALIDATION,),
        input_hashes={"validation": validation_hash},
    )
    return best, tuple(trials), record
