import json
from pathlib import Path

import pytest

from fuzzyxai.core.alignment import compute_gamma, compute_gamma_route
from fuzzyxai.core.proof_package import build_proof_package, verify_proof_package
from fuzzyxai.core.reduction import compute_reduction_loss
from fuzzyxai.core.risk_observer import compute_risk
from fuzzyxai.core.scenario_engine import DEFAULT_HYBRID_PLAN, compute_hybrid_xiris, hybrid_xiris_engine_payload
from fuzzyxai.export_tables import export_hybrid_xiris_tables
from fuzzyxai.run_scenario import run_hybrid_xiris_batch
from fuzzyxai.studio.operator_scenarios import build_report, load_scenarios


def test_hybrid_alignment_gamma_formula() -> None:
    assert compute_gamma(
        components={"d_mu": 0.39, "d_R": 0.40, "d_u": 0.5675, "d_tau": 0.0},
        weights={"d_mu": 0.25, "d_R": 0.35, "d_u": 0.20, "d_tau": 0.20},
    ) == pytest.approx(0.351)


def test_gis_route_gamma_formula() -> None:
    assert compute_gamma_route(
        p=0.67,
        alpha_mean=0.72,
        feature_support=0.47,
        mode="model_to_explanations",
    ) == pytest.approx(0.20)


def test_hybrid_reduction_delta_formula() -> None:
    assert compute_reduction_loss(
        DEFAULT_HYBRID_PLAN.reduction_components,
        DEFAULT_HYBRID_PLAN.reduction_weights,
    ) == pytest.approx(0.106811)


def test_hybrid_risk_formula() -> None:
    assert compute_risk(
        components={"model_signal": 0.88, "block_rule": 0.91, "source_conflict": 1.0, "reduction_component": 0.3225},
        weights={"model_signal": 0.35, "block_rule": 0.25, "source_conflict": 0.20, "reduction_component": 0.20},
    ) == pytest.approx(0.800)


def test_hybrid_engine_computes_block_from_inputs() -> None:
    scenario = next(s for s in load_scenarios() if s["scenario_id"] == "hybrid_xiris")
    result = compute_hybrid_xiris()

    assert result.gamma == pytest.approx(scenario["expected_result"]["gamma"])
    assert result.delta == pytest.approx(scenario["expected_result"]["delta"])
    assert result.rho == pytest.approx(scenario["expected_result"]["rho"])
    assert result.chi_r_crit == 1
    assert result.action == scenario["expected_result"]["action"]
    assert result.occurrences == ["alignment", "risk_observer", "action"]
    assert result.diagnostic_id == "D_quality_source_conflict"


def test_proof_package_build_and_verify() -> None:
    scenario = next(s for s in load_scenarios() if s["scenario_id"] == "hybrid_xiris")
    report = build_report(scenario)
    package = build_proof_package(scenario, report, hybrid_xiris_engine_payload())
    verification = verify_proof_package(package)

    assert verification.valid
    assert verification.errors == []
    assert package["final_action"] == "block"
    assert package["computed_result"]["rho"] == pytest.approx(0.800)
    values = {item["node_id"]: item["computed"] for item in package["operator_values"]}
    assert values["alignment"]["gamma_ij"] == pytest.approx(package["computed_result"]["gamma"])
    assert values["reduction"]["delta"] == pytest.approx(package["computed_result"]["delta"])
    assert values["risk_observer"]["rho"] == pytest.approx(package["computed_result"]["rho"])
    assert values["action"]["action"] == package["computed_result"]["action"]


def test_proof_package_verifier_rejects_inconsistent_operator_trace() -> None:
    scenario = next(s for s in load_scenarios() if s["scenario_id"] == "hybrid_xiris")
    package = build_proof_package(scenario, build_report(scenario), hybrid_xiris_engine_payload())
    package["operator_values"][0]["computed"]["gamma_ij"] = 0.35
    package["package_hash"] = "invalid-after-edit"
    verification = verify_proof_package(package)

    assert not verification.valid
    assert any(error.startswith("inconsistent:alignment.gamma") for error in verification.errors)


def test_engine_does_not_read_expected_result() -> None:
    scenario = next(s for s in load_scenarios() if s["scenario_id"] == "hybrid_xiris")
    scenario["expected_result"] = {"gamma": 999, "delta": 999, "rho": 999, "action": "accept"}
    result = compute_hybrid_xiris()

    assert result.gamma == pytest.approx(0.351)
    assert result.delta == pytest.approx(0.106811)
    assert result.rho == pytest.approx(0.800)
    assert result.action == "block"


def test_hybrid_batch_run_reproduces_summary(tmp_path: Path) -> None:
    summary = run_hybrid_xiris_batch(tmp_path)

    assert summary["n_cases"] == 1000
    assert summary["accept"] == 612
    assert summary["lower_confidence"] == 201
    assert summary["block"] == 187
    assert summary["baseline_critical_misses"] == 168
    assert summary["fuzzyxai_critical_misses"] == 0
    assert summary["proof_package_valid"] is True
    cases = (tmp_path / "hybrid_xiris_batch_cases.csv").read_text(encoding="utf-8").splitlines()
    assert cases[0] == "case_id,scenario_id,is_critical_case,baseline_action,fuzzyxai_action,baseline_miss,fuzzyxai_miss"

    saved = json.loads((tmp_path / "hybrid_xiris_batch_summary.json").read_text(encoding="utf-8"))
    assert saved == summary


def test_exported_tables_include_explainplan_and_risk_decomposition(tmp_path: Path) -> None:
    paths = export_hybrid_xiris_tables(tmp_path)
    names = {path.name for path in paths}

    assert "table_5_2_explainplan.csv" in names
    assert "table_5_6_risk_decomposition.csv" in names
    explainplan = (tmp_path / "table_5_2_explainplan.csv").read_text(encoding="utf-8")
    risk = (tmp_path / "table_5_6_risk_decomposition.csv").read_text(encoding="utf-8")
    assert "alignment,w_d_mu,0.25" in explainplan
    assert "risk,w_source_conflict,0.2" in explainplan
    assert "total,rho,,0.8" in risk
