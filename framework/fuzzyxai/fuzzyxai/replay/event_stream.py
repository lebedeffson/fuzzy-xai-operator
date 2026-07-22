"""Deterministic, streaming operational replay events."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterator, Sequence

import numpy as np


DEFAULT_PHASES = (
    "clean",
    "gradual_shift",
    "sudden_shift",
    "stale_calibration",
    "model_checkpoint_change",
    "wrong_preprocessing",
    "partial_provenance_loss",
    "reference_population_error",
    "explainer_instability",
    "double_faults",
    "triple_faults",
    "recovery",
)


@dataclass(frozen=True)
class ReplayEvent:
    event_id: str
    sequence_index: int
    phase: str
    modality: str
    confidence: float
    predictive_error_probability: float
    route_faults: tuple[str, ...]
    explanation_instability: float
    shift_score: float
    delayed_label: bool
    hard_fault: bool


def stream_events(
    count: int = 500_000,
    *,
    seed: int = 4201,
    phases: Sequence[str] = DEFAULT_PHASES,
) -> Iterator[ReplayEvent]:
    if count <= 0 or not phases:
        raise ValueError("positive event count and phases are required")
    rng = np.random.default_rng(seed)
    modalities = np.asarray(("tabular", "image", "text", "time_series"))
    for index in range(count):
        phase_index = min(len(phases) - 1, index * len(phases) // count)
        phase = phases[phase_index]
        progress = (index * len(phases) % count) / count
        shift, instability, faults = _phase_state(phase, progress, rng)
        base_error = 0.02 + 0.28 * shift + 0.12 * instability
        error_probability = float(np.clip(base_error + rng.normal(0.0, 0.01), 0.001, 0.95))
        confidence = float(np.clip(1.0 - error_probability + rng.normal(0.0, 0.08), 0.01, 0.999))
        yield ReplayEvent(
            event_id=f"replay-{index:09d}",
            sequence_index=index,
            phase=phase,
            modality=str(modalities[index % len(modalities)]),
            confidence=confidence,
            predictive_error_probability=error_probability,
            route_faults=faults,
            explanation_instability=instability,
            shift_score=shift,
            delayed_label=bool(rng.random() < error_probability),
            hard_fault=any(item in {"model_checkpoint_change", "wrong_preprocessing", "canonical_corruption"} for item in faults),
        )


def _phase_state(phase: str, progress: float, rng: np.random.Generator) -> tuple[float, float, tuple[str, ...]]:
    if phase == "clean":
        return 0.02, 0.03, ()
    if phase == "gradual_shift":
        return 0.05 + 0.55 * progress, 0.08, ()
    if phase == "sudden_shift":
        return 0.75, 0.15, ()
    if phase == "stale_calibration":
        return 0.35, 0.10, ("stale_calibration",)
    if phase == "model_checkpoint_change":
        return 0.25, 0.15, ("model_checkpoint_change",)
    if phase == "wrong_preprocessing":
        return 0.35, 0.20, ("wrong_preprocessing",)
    if phase == "partial_provenance_loss":
        return 0.12, 0.20, ("partial_provenance_loss",)
    if phase == "reference_population_error":
        return 0.40, 0.15, ("reference_population_error",)
    if phase == "explainer_instability":
        return 0.20, 0.85, ("explainer_instability",)
    if phase == "double_faults":
        return 0.45, 0.55, ("partial_provenance_loss", "stale_calibration")
    if phase == "triple_faults":
        return 0.65, 0.75, ("model_checkpoint_change", "reference_population_error", "canonical_corruption")
    if phase == "recovery":
        residual = max(0.02, 0.50 * (1.0 - progress))
        return residual, residual, () if progress > 0.2 else ("stale_calibration",)
    return float(rng.uniform(0.0, 0.3)), float(rng.uniform(0.0, 0.3)), ()
