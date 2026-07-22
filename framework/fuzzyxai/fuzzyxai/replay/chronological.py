"""Chronological multi-model replay with burst incidents and partial recovery."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterator, Sequence

import numpy as np


@dataclass(frozen=True)
class IncidentWindow:
    incident_id: str
    start: int
    duration: int
    fault_family: str
    model_lane: str
    repairable: bool
    repair_success_probability: float

    @property
    def stop(self) -> int:
        return self.start + self.duration


@dataclass(frozen=True)
class ChronologicalEvent:
    event_id: str
    timestamp_index: int
    model_lane: str
    confidence: float
    drift_score: float
    active_incidents: tuple[str, ...]
    repairable: bool
    delayed_model_error: bool
    label_delay: int
    recurrence: bool


def registered_incident_schedule(count: int, *, seed: int = 5201) -> tuple[IncidentWindow, ...]:
    if count < 10_000:
        raise ValueError("chronological replay requires at least 10,000 events")
    rng = np.random.default_rng(seed)
    families = (
        ("stale_calibration", True, 0.98),
        ("partial_preprocessing_update", True, 0.95),
        ("reference_population_error", True, 0.90),
        ("explainer_model_mismatch", True, 0.97),
        ("model_checkpoint_corruption", False, 0.0),
    )
    lanes = ("model-a", "model-b", "model-c")
    incidents = []
    cursor = int(0.12 * count)
    for index in range(14):
        family, repairable, probability = families[index % len(families)]
        gap = int(rng.integers(max(100, count // 80), max(200, count // 25)))
        cursor = min(count - 100, cursor + gap)
        duration = int(rng.integers(max(50, count // 500), max(100, count // 120)))
        incidents.append(IncidentWindow(f"incident-{index:03d}", cursor, min(duration, count - cursor), family, lanes[index % len(lanes)], repairable, probability))
    # A later recurrence is deliberately separated from the first occurrence.
    first = incidents[1]
    recurrence_start = min(count - 100, int(0.87 * count))
    incidents.append(IncidentWindow("incident-recurrence", recurrence_start, min(first.duration // 2, count - recurrence_start), first.fault_family, first.model_lane, True, 0.80))
    return tuple(sorted(incidents, key=lambda item: (item.start, item.incident_id)))


def stream_chronological_events(
    count: int = 500_000,
    *,
    seed: int = 5201,
    incidents: Sequence[IncidentWindow] | None = None,
) -> Iterator[ChronologicalEvent]:
    schedule = tuple(incidents or registered_incident_schedule(count, seed=seed))
    rng = np.random.default_rng(seed)
    lanes = ("model-a", "model-b", "model-c")
    previous_families: set[str] = set()
    drift = 0.01
    for index in range(count):
        lane = lanes[index % len(lanes)]
        if int(0.25 * count) <= index < int(0.45 * count):
            drift = min(0.55, drift + 2.7 / count)
        elif int(0.70 * count) <= index:
            drift = max(0.03, drift - 1.8 / count)
        active = tuple(item for item in schedule if item.start <= index < item.stop and item.model_lane == lane)
        families = tuple(item.fault_family for item in active)
        recurrence = any(item.fault_family in previous_families for item in active)
        previous_families.update(families)
        repairable = bool(active) and all(item.repairable for item in active)
        error_probability = float(np.clip(0.015 + 0.22 * drift + 0.18 * len(active), 0.001, 0.90))
        confidence = float(np.clip(1.0 - error_probability + rng.normal(0.0, 0.05), 0.01, 0.999))
        yield ChronologicalEvent(
            event_id=f"chronological-{index:09d}",
            timestamp_index=index,
            model_lane=lane,
            confidence=confidence,
            drift_score=float(drift),
            active_incidents=families,
            repairable=repairable,
            delayed_model_error=bool(rng.random() < error_probability),
            label_delay=int(rng.integers(25, 501)),
            recurrence=recurrence,
        )
