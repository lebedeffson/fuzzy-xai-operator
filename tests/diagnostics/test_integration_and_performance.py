from __future__ import annotations

import pytest

from experiments.diagnostic_v21.benchmark import PIPELINES, route_fixture, run
from experiments.diagnostic_v21.protocol_gate import validate_draft
from fuzzyxai import FuzzyXAI


@pytest.mark.parametrize(("pipeline_id", "modality"), PIPELINES)
def test_six_pipeline_families_use_the_same_public_api(pipeline_id: str, modality: str) -> None:
    valid = FuzzyXAI().diagnose(route=route_fixture(pipeline_id, modality, 0))
    invalid = FuzzyXAI().diagnose(route=route_fixture(pipeline_id, modality, 1))
    assert valid.route_status == "valid"
    assert invalid.route_status == "invalid"
    assert invalid.minimal_cut is not None


def test_diagnostic_p95_is_below_one_second(tmp_path) -> None:
    result = run(repetitions=5, output=tmp_path / "performance.json")
    assert result["status"] == "PASS"
    assert result["timings"]["total"]["p95_ms"] < 1000


def test_h10_c2_protocol_remains_preconfirmatory() -> None:
    result = validate_draft("config/h10_c2_diagnostic_cut_protocol.yaml")
    assert result["status"] == "PASS"
    assert result["scientific_status"] == "BLOCKED_PRECONFIRMATORY"
