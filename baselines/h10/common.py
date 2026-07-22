from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class BaselineResult:
    route_status: str
    parent_family: str | None = None
    leaf_type: str | None = None
    source_nodes: tuple[str, ...] = ()
    cut_nodes: tuple[str, ...] = ()
    repair_nodes: tuple[str, ...] = ()
    unknown: bool = False
    abstained: bool = False
    confidence: float = 0.0
    anomaly_score: float = 0.0


def fields(route: Any) -> tuple[str, ...]:
    return tuple(sorted(set(route.expected) | set(route.observed)))


def missing(route: Any) -> tuple[str, ...]:
    return tuple(sorted(field for field in route.mandatory_fields if route.observed.get(field) in (None, "", (), [])))


def changed(route: Any) -> tuple[str, ...]:
    return tuple(sorted(field for field in fields(route) if field in route.observed and route.expected.get(field) != route.observed.get(field)))
