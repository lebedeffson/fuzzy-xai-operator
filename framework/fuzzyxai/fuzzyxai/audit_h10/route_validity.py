from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .models import RouteObservation


@dataclass(frozen=True)
class ValidityResult:
    status: str
    mismatched_fields: tuple[str, ...]
    missing_fields: tuple[str, ...]
    invalid_paths: tuple[frozenset[str], ...]


def _different(expected: Any, observed: Any, field: str) -> bool:
    if field == "calibration_age_days":
        return float(observed or 0.0) > float(expected or 30.0)
    if field == "reduction_loss":
        return float(observed or 0.0) > float(expected or 0.1)
    return expected != observed


def validate_route(route: RouteObservation) -> ValidityResult:
    missing = tuple(sorted(field for field in route.mandatory_fields if route.observed.get(field) in (None, "", (), [])))
    mismatches = tuple(
        sorted(
            field
            for field, expected in route.expected.items()
            if field not in missing and field in route.observed and _different(expected, route.observed[field], field)
        )
    )
    if missing:
        status = "insufficient_evidence"
    elif mismatches:
        status = "invalid"
    else:
        status = "valid"
    affected = set(missing) | set(mismatches)
    paths = tuple(frozenset(node for node in path if node in affected) for path in route.dependency_paths)
    paths = tuple(path for path in paths if path)
    if affected and not paths:
        paths = tuple(frozenset((field,)) for field in sorted(affected))
    return ValidityResult(status, mismatches, missing, paths)
