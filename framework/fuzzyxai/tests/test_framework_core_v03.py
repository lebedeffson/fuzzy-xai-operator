from pathlib import Path

import pytest

from fuzzyxai import build_proof_trace, build_route, render_dashboard, save_route_json, verify_proof_trace
from fuzzyxai.examples.hybrid_xiris import get_input
from fuzzyxai.viz import save_proof_trace_json


def test_public_framework_api_builds_hybrid_route():
    route = build_route(get_input())
    result = route.computed_result

    assert result["gamma"] == pytest.approx(0.351)
    assert result["delta"] == pytest.approx(0.106811)
    assert result["r_delta"] == pytest.approx(0.3225)
    assert result["rho"] == pytest.approx(0.800)
    assert result["chi_crit"] == 1
    assert result["diagnostic_id"] == "D_quality_source_conflict"
    assert result["action"] == "block"
    assert route.final_action == "block"


def test_proof_trace_verifier_passes_and_rejects_tamper():
    route = build_route(get_input())
    trace = build_proof_trace(route)
    assert verify_proof_trace(trace).valid

    tampered = trace.to_dict()
    tampered["computed_result"]["action"] = "accept"
    assert not verify_proof_trace(tampered).valid


def test_framework_exports_route_proof_and_dashboard(tmp_path: Path):
    route = build_route(get_input())
    trace = build_proof_trace(route)

    route_path = save_route_json(route, tmp_path / "route.json")
    proof_path = save_proof_trace_json(trace, tmp_path / "proof_trace.json")
    figure_path = render_dashboard(route, tmp_path / "operator_dashboard.png")

    assert route_path.exists()
    assert proof_path.exists()
    assert figure_path.exists()
    assert figure_path.stat().st_size > 0
