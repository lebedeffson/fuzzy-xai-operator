import json
from pathlib import Path

import pytest

from fuzzyxai.core.scenario_engine import compute_hybrid_xiris, hybrid_xiris_engine_payload
from fuzzyxai.studio.operator_scenarios import build_report, load_scenarios


def _scenario() -> dict:
    return next(s for s in load_scenarios() if s["scenario_id"] == "hybrid_xiris")


def test_proof_package_matches_engine_result() -> None:
    package = json.loads(Path("reports/studio_batch/hybrid_xiris_proof_package.json").read_text(encoding="utf-8"))
    result = compute_hybrid_xiris()
    values = {item["node_id"]: item["computed"] for item in package["operator_values"]}

    assert package["computed_result"]["gamma"] == pytest.approx(result.gamma)
    assert package["computed_result"]["delta"] == pytest.approx(result.delta)
    assert package["computed_result"]["rho"] == pytest.approx(result.rho)
    assert package["computed_result"]["action"] == result.action
    assert values["alignment"]["gamma_ij"] == pytest.approx(result.gamma)
    assert values["reduction"]["delta"] == pytest.approx(result.delta)
    assert values["risk_observer"]["rho"] == pytest.approx(result.rho)


def test_exported_tables_match_proof_package() -> None:
    package = json.loads(Path("reports/studio_batch/hybrid_xiris_proof_package.json").read_text(encoding="utf-8"))
    summary = Path("reports/chapter5/studio_tables/table_5_5_run_summary.csv").read_text(encoding="utf-8")
    risk = Path("reports/chapter5/studio_tables/table_5_6_risk_decomposition.csv").read_text(encoding="utf-8")

    assert f"gamma,{package['computed_result']['gamma']}" in summary
    assert f"delta,{package['computed_result']['delta']}" in summary
    assert f"rho,{package['computed_result']['rho']}" in summary
    assert f"rho,total,,{package['computed_result']['rho']}" in risk


def test_studio_trace_matches_engine_result() -> None:
    report = build_report(_scenario())
    engine = hybrid_xiris_engine_payload()["computed_result"]
    operators = {item["node_id"]: item["computed"] for item in report["operators"]}

    assert operators["alignment"]["gamma_ij"] == pytest.approx(engine["gamma"])
    assert operators["reduction"]["delta"] == pytest.approx(engine["delta"])
    assert operators["risk_observer"]["rho"] == pytest.approx(engine["rho"])
    assert operators["action"]["action"] == engine["action"]


def test_expected_result_is_not_engine_input() -> None:
    scenario = _scenario()
    scenario["expected_result"]["rho"] = 0.123
    result = compute_hybrid_xiris()

    assert result.rho == pytest.approx(0.800)
    assert result.action == "block"
