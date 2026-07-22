from __future__ import annotations

from experiments.h10.routes import build_route
from experiments.h10.run_replay import _drift_variant
from fuzzyxai.audit_h10.route_validity import validate_route


def test_normal_drift_changes_context_jointly_without_contract_fault() -> None:
    route = build_route("clean", "tabular", "object-1")
    drifted = _drift_variant(route, 7)
    assert drifted.expected == drifted.observed
    assert validate_route(drifted).status == "valid"
