"""Registered H6 detectability grid and eligible-region summaries."""

from __future__ import annotations

from dataclasses import dataclass
from itertools import product
from typing import Iterable


@dataclass(frozen=True)
class DetectabilityPoint:
    effect_strength: float
    support: float
    redundancy: float
    noise: str
    interaction_order: int
    proxy_correlation: float

    @property
    def eligible(self) -> bool:
        return self.effect_strength >= 0.05 and self.support >= 0.05 and self.redundancy <= 0.50


def registered_detectability_grid() -> tuple[DetectabilityPoint, ...]:
    return tuple(
        DetectabilityPoint(*values)
        for values in product(
            (0.01, 0.02, 0.05, 0.10, 0.20),
            (0.01, 0.025, 0.05, 0.10, 0.25),
            (0.0, 0.25, 0.50, 0.75, 0.90),
            ("low", "medium", "high"),
            (1, 2, 3),
            (0.0, 0.3, 0.6, 0.9),
        )
    )


def summarize_detectability(rows: Iterable[tuple[DetectabilityPoint, bool, bool, bool]]) -> dict[str, float | int | bool]:
    selected = [(point, detected, false_discovery, sign_correct) for point, detected, false_discovery, sign_correct in rows if point.eligible]
    if not selected:
        raise ValueError("eligible-region observations are required")
    detection = sum(item[1] for item in selected) / len(selected)
    fdr = sum(item[2] for item in selected) / len(selected)
    sign = sum(item[3] for item in selected) / len(selected)
    return {
        "n_eligible": len(selected),
        "detection_rate": detection,
        "false_discovery_rate": fdr,
        "sign_accuracy": sign,
        "criterion_met": detection >= 0.85 and fdr <= 0.10 and sign >= 0.90,
    }
