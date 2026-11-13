"""Regression: legacy sample_113 does not match its hashed ExplainPlan.

The assertions preserve the historical mismatch as a reproducible
finding; they are not assertions that the experiment was plan-compliant.
"""
from __future__ import annotations

import pytest

from experiments.paper_chapter2.audit_sample113 import audit


def test_legacy_report_conformance_is_not_falsely_certified() -> None:
    result = audit()
    assert result["id"] == "sample_113"
    assert result["plan_hash"] == "b3c3fc83fbba3f862e06470ddd1fb9ae77bb57e39c14575534275049e669209e"
    assert result["checks"] == {
        "hash_matches_stated_yaml": True,
        "source_weights_reproduce_archived_uncertainty": True,
        "stated_yaml_weights_reproduce_archived_uncertainty": False,
        "stated_yaml_memberships_match_archive": False,
        "stated_yaml_required_trace_fields_exist": False,
    }
    assert result["contract_conformant"] is False
    assert result["weights"]["source"] == {"u_model": 0.4, "u_rules": 0.4, "u_trace": 0.2}
    assert result["weights"]["yaml"] == {"model": 0.5, "rules": 0.3, "trace": 0.2}
    assert result["uncertainty"]["observed"] == pytest.approx(0.35970753235352704)
    assert result["uncertainty"]["stated_yaml"] == pytest.approx(0.38934982671632096)
    assert result["risk_membership"]["archive"]["high"] == pytest.approx(0.7035770563720608)
    assert result["risk_membership"]["yaml"]["high"] == pytest.approx(0.8143082254882432)
    assert result["risk_membership"]["yaml"]["low"] == pytest.approx(0.0)
    assert result["missing_trace_fields"] == ["time"]
