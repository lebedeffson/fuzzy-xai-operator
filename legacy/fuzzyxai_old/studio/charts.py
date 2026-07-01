from __future__ import annotations

from typing import Any


def route_rows(case_state: dict[str, Any]) -> list[dict[str, Any]]:
    route = case_state.get('route_header', {})
    return [{'step': k, 'state': v} for k, v in route.items()]


def representation_rows(case_state: dict[str, Any]) -> list[dict[str, Any]]:
    return list(case_state.get('uncertainty', {}).get('classes', []))


def composition_rows(case_state: dict[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for edge in case_state.get('composition', {}).get('edges', []):
        rows.append(
            {
                'transition': f"{edge.get('source')} -> {edge.get('target')}",
                'gamma': edge.get('gamma'),
                'gamma_max': edge.get('gamma_max'),
                'status': edge.get('status'),
                'hott': edge.get('hott'),
            }
        )
    return rows
