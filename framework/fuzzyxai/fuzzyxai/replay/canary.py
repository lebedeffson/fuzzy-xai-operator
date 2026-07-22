"""Deterministic canary assignment and staged rollout."""

from __future__ import annotations

import hashlib


CANARY_STAGES = (0.05, 0.10, 0.25, 1.0)


def in_canary(event_id: str, fraction: float) -> bool:
    if fraction not in CANARY_STAGES:
        raise ValueError("canary fraction must be 5%, 10%, 25%, or 100%")
    bucket = int(hashlib.sha256(event_id.encode()).hexdigest()[:12], 16) / float(16**12)
    return bucket < fraction
