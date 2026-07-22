from __future__ import annotations

from dataclasses import asdict
from typing import Any

from fuzzyxai.audit_h10.models import RouteObservation

from .mutations import MutationTruth
from .oracle_v19 import from_dict as oracle_truth_from_dict, to_dict as oracle_truth_to_dict


def route_to_dict(route: RouteObservation) -> dict[str, Any]:
    return asdict(route)


def route_from_dict(payload: dict[str, Any]) -> RouteObservation:
    return RouteObservation(
        route_id=payload["route_id"],
        dataset_id=payload["dataset_id"],
        modality=payload["modality"],
        object_id=payload["object_id"],
        expected=payload["expected"],
        observed=payload["observed"],
        mandatory_fields=tuple(payload["mandatory_fields"]),
        dependency_paths=tuple(tuple(path) for path in payload["dependency_paths"]),
        repair_costs={key: float(value) for key, value in payload["repair_costs"].items()},
    )


def truth_to_dict(truth: MutationTruth) -> dict[str, Any]:
    return oracle_truth_to_dict(truth)


def truth_from_dict(payload: dict[str, Any]) -> MutationTruth:
    return oracle_truth_from_dict(payload)
