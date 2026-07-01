from pathlib import Path

import pytest

from fuzzyxai import build_proof_trace, build_route, render_dashboard, verify_proof_trace
from fuzzyxai.examples import list_examples, load_example


EXPECTED = {
    "hybrid_xiris": {
        "action": "block",
        "diagnostic": "D_quality_source_conflict",
        "gamma": 0.351,
        "delta": 0.106811,
        "r_delta": 0.3225,
        "rho": 0.800,
        "chi_crit": 1,
    },
    "medical_ecg_signal": {
        "action": "defer_to_human",
        "diagnostic": "D_signal_quality",
        "quality_score": 0.58,
        "missing_fragments": 2,
    },
    "gd_anfis_shap": {
        "action": "audit",
        "diagnostic": "D_rule_attribution_conflict",
        "gamma_rule_shap": 0.685,
    },
    "beacon_xai": {
        "action": "audit",
        "diagnostic": "D_counterevidence_conflict",
        "counter_fragments": 30,
        "objects_with_counterevidence": 83,
    },
    "gis_integro": {
        "action": "audit_report",
        "diagnostic": "D_route_context_limit",
        "gamma_route": 0.20,
        "delta": 0.08,
        "formula": "max(|p - alpha_mean|, |p - s|)",
    },
}


def test_all_examples_are_registered():
    assert set(list_examples()) == set(EXPECTED)


@pytest.mark.parametrize("scenario_id", sorted(EXPECTED))
def test_all_scenarios_build_routes_and_verify(scenario_id: str, tmp_path: Path):
    route = build_route(load_example(scenario_id))
    trace = build_proof_trace(route)
    verification = verify_proof_trace(trace)
    expected = EXPECTED[scenario_id]

    assert verification.valid
    assert route.scenario_id == scenario_id
    assert route.final_action == expected["action"]
    assert route.computed_result["action"] == expected["action"]
    assert route.computed_result["diagnostic_id"] == expected["diagnostic"]
    assert route.verifier_status == "PASS"
    assert len(route.nodes) >= 5

    for key, value in expected.items():
        if key in {"action", "diagnostic"}:
            continue
        if isinstance(value, float):
            assert route.computed_result[key] == pytest.approx(value)
        else:
            assert route.computed_result[key] == value

    figure = render_dashboard(route, tmp_path / f"{scenario_id}.png")
    assert figure.exists()
    assert figure.stat().st_size > 0
