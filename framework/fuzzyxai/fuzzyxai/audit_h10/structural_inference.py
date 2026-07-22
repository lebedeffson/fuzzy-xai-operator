from __future__ import annotations

from itertools import combinations

from .models import FaultPrediction
from .taxonomy import FAULT_SPECS, SPEC_BY_LEAF, FaultSpec


def infer_specs(active_fields: tuple[str, ...], fault: FaultPrediction | None = None) -> tuple[FaultSpec, ...]:
    active = set(active_fields)
    if not active:
        return ()
    candidates = tuple(spec for spec in FAULT_SPECS if active & set(spec.fields))
    if not candidates:
        return ()
    anchor = SPEC_BY_LEAF.get(fault.leaf_type) if fault is not None else None
    best: tuple[tuple[float, ...], tuple[FaultSpec, ...]] | None = None
    max_size = min(3, len(candidates))
    for size in range(1, max_size + 1):
        for combo in combinations(candidates, size):
            union = set().union(*(set(spec.fields) for spec in combo))
            covered = len(active & union)
            coverage = covered / len(active)
            precision = covered / max(len(union), 1)
            jaccard = covered / max(len(active | union), 1)
            anchor_bonus = 1.0 if anchor is not None and anchor in combo else 0.0
            # Prefer complete coverage, then compact type explanations, then the
            # classifier-supported leaf. Lexicographic scoring is deterministic.
            score = (coverage, jaccard, precision, anchor_bonus, -float(size))
            if best is None or score > best[0] or (score == best[0] and tuple(spec.leaf for spec in combo) < tuple(spec.leaf for spec in best[1])):
                best = (score, combo)
    assert best is not None
    return tuple(best[1])
