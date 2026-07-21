"""Deterministic validation-only calibration for decision policies."""

from __future__ import annotations

import itertools
from dataclasses import asdict, dataclass
from typing import Mapping, Sequence

from .metrics import decision_policy_metrics
from .policies import PolicySignals, apply_policy


@dataclass(frozen=True)
class CalibrationTrial:
    confidence_threshold: float
    conflict_threshold: float
    stability_threshold: float
    risk: float
    critical_wrong_automatic: int
    false_blocks: int
    reviewed: int
    representation_cost: float
    runtime_cost: float

    def tie_break_key(self) -> tuple[float, int, float, int, tuple[float, float, float]]:
        return (
            self.risk,
            self.critical_wrong_automatic,
            self.representation_cost,
            3,
            (self.confidence_threshold, self.conflict_threshold, self.stability_threshold),
        )

    def to_dict(self) -> dict[str, float | int]:
        return asdict(self)


def deterministic_grid_search(
    *,
    labels: Sequence[int],
    predictions: Sequence[int],
    critical_mask: Sequence[bool],
    signals: PolicySignals,
    costs: Mapping[str, float],
    confidence_grid: Sequence[float],
    conflict_grid: Sequence[float],
    stability_grid: Sequence[float],
    representation_cost: float = 0.0,
    runtime_cost: float = 0.0,
) -> tuple[CalibrationTrial, list[CalibrationTrial]]:
    trials: list[CalibrationTrial] = []
    for confidence, conflict, stability in itertools.product(confidence_grid, conflict_grid, stability_grid):
        actions = apply_policy(
            "P5",
            signals,
            confidence_threshold=float(confidence),
            conflict_threshold=float(conflict),
            stability_threshold=float(stability),
        )
        metrics = decision_policy_metrics(labels, predictions, actions, critical_mask, costs=dict(costs))
        risk = float(metrics["risk"]) + representation_cost + runtime_cost
        trials.append(
            CalibrationTrial(
                confidence_threshold=float(confidence),
                conflict_threshold=float(conflict),
                stability_threshold=float(stability),
                risk=risk,
                critical_wrong_automatic=int(metrics["critical_wrong_automatic"]),
                false_blocks=int(metrics["false_blocks"]),
                reviewed=int(metrics["reviewed"]),
                representation_cost=float(representation_cost),
                runtime_cost=float(runtime_cost),
            )
        )
    return min(trials, key=CalibrationTrial.tie_break_key), trials


def freeze_calibration(
    best: CalibrationTrial,
    *,
    dataset_id: str,
    split_hash: str,
    code_commit: str,
    seed: int,
    library_versions: Mapping[str, str],
) -> dict[str, object]:
    return {
        "schema_version": "1.0",
        "status": "frozen_from_validation",
        "test_partition_used": False,
        "dataset_id": dataset_id,
        "split_hash": split_hash,
        "code_commit": code_commit,
        "seed": seed,
        "library_versions": dict(library_versions),
        "best_config": best.to_dict(),
        "tie_break": [
            "lower critical-error risk",
            "lower representation complexity",
            "fewer calibrated parameters",
            "lexicographically first configuration",
        ],
    }
