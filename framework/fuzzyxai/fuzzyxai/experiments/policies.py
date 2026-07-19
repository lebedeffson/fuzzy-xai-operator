"""Predeclared decision policies and risk-cost scenarios."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Sequence

import numpy as np


PREDECLARED_COSTS: dict[str, dict[str, float]] = {
    "critical_errors_expensive": {"critical_error": 25.0, "wrong_auto": 5.0, "review": 1.0, "false_block": 2.0},
    "balanced": {"critical_error": 10.0, "wrong_auto": 4.0, "review": 1.5, "false_block": 2.5},
    "manual_review_expensive": {"critical_error": 12.0, "wrong_auto": 4.0, "review": 5.0, "false_block": 3.0},
}


@dataclass(frozen=True)
class PolicySignals:
    confidence: np.ndarray
    shap_support: np.ndarray
    lime_support: np.ndarray
    explanation_stability: np.ndarray
    critical_rupture: np.ndarray
    history_instability: np.ndarray

    def __post_init__(self) -> None:
        lengths = {len(value) for value in self.__dict__.values()}
        if len(lengths) != 1:
            raise ValueError("policy signal arrays must align")


def apply_policy(
    policy_id: str,
    signals: PolicySignals,
    *,
    confidence_threshold: float = 0.75,
    conflict_threshold: float = 0.25,
    stability_threshold: float = 0.65,
) -> np.ndarray:
    confidence = np.asarray(signals.confidence, dtype=float)
    conflict = np.abs(np.asarray(signals.shap_support) - np.asarray(signals.lime_support))
    stable = np.asarray(signals.explanation_stability) >= stability_threshold
    rupture = np.asarray(signals.critical_rupture, dtype=bool)
    history = np.asarray(signals.history_instability, dtype=bool)
    actions = np.full(len(confidence), "review", dtype="U6")
    if policy_id == "P1":
        actions[confidence >= confidence_threshold] = "accept"
    elif policy_id == "P2":
        actions[(confidence >= confidence_threshold) & (signals.shap_support >= 0.5)] = "accept"
    elif policy_id == "P3":
        actions[(confidence >= confidence_threshold) & (conflict <= conflict_threshold)] = "accept"
    elif policy_id == "P4":
        actions[(confidence >= confidence_threshold) & stable & ~rupture] = "accept"
        actions[rupture] = "block"
    elif policy_id == "P5":
        actions[(confidence >= confidence_threshold) & stable & ~rupture & ~history] = "accept"
        actions[rupture] = "block"
    elif policy_id == "P6":
        pass
    elif policy_id == "P7":
        actions[:] = "accept"
    else:
        raise ValueError(f"unknown policy: {policy_id}")
    return actions
