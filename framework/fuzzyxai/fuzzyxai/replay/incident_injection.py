"""Registered incident injection for controlled replay."""

from __future__ import annotations

from dataclasses import replace
from typing import Sequence

from .event_stream import ReplayEvent


REGISTERED_INCIDENTS = frozenset(
    {
        "stale_calibration",
        "model_checkpoint_change",
        "wrong_preprocessing",
        "partial_provenance_loss",
        "reference_population_error",
        "explainer_instability",
        "canonical_corruption",
    }
)


def inject_incidents(event: ReplayEvent, incidents: Sequence[str]) -> ReplayEvent:
    unknown = set(incidents) - REGISTERED_INCIDENTS
    if unknown:
        raise ValueError(f"unregistered replay incidents: {sorted(unknown)}")
    faults = tuple(dict.fromkeys((*event.route_faults, *incidents)))
    hard = event.hard_fault or any(item in {"model_checkpoint_change", "wrong_preprocessing", "canonical_corruption"} for item in incidents)
    return replace(event, route_faults=faults, hard_fault=hard)
