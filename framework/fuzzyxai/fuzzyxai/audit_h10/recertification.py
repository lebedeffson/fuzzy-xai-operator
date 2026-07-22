from __future__ import annotations

from dataclasses import replace

from .models import RepairAction, RouteObservation
from .route_validity import validate_route


def execute_and_recertify(route: RouteObservation, actions: tuple[RepairAction, ...]) -> tuple[RouteObservation, bool]:
    observed = dict(route.observed)
    for action in actions:
        fields = action.affected_fields or ((action.target,) if action.target in route.expected else ())
        for field in fields:
            if field in route.expected:
                observed[field] = route.expected[field]
    repaired = replace(route, observed=observed)
    return repaired, validate_route(repaired).status == "valid"
