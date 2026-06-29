from __future__ import annotations

from apps.layered_demo import _route_statuses


def test_layered_route_statuses_smoke() -> None:
    state = {'route_header': {'Input': 'built', 'Action': 'block'}}
    values = _route_statuses(state)
    assert 'built' in values
    assert 'block' in values
