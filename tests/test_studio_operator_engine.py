import json
from pathlib import Path

import pytest

from fuzzyxai.core.alignment import compute_gamma, compute_gamma_route
from fuzzyxai.core.proof_package import build_proof_package, verify_proof_package
from fuzzyxai.core.reduction import compute_reduction_loss
from fuzzyxai.core.risk_observer import compute_risk
from fuzzyxai.core.scenario_engine import DEFAULT_HYBRID_PLAN, compute_hybrid_xiris, hybrid_xiris_engine_payload
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


def test_proof_package_build_and_verify() -> None:
    scenario = next(s for s in load_scenarios() if s["scenario_id"] == "hybrid_xiris")
    report = build_report(scenario)
    package = build_proof_package(scenario, report, hybrid_xiris_engine_payload())
    verification = verify_proof_package(package)

    assert verification.valid
    assert verification.errors == []
    assert package["final_action"] == "block"
    assert package["computed_result"]["rho"] == pytest.approx(0.800)


def test_hybrid_batch_run_reproduces_summary(tmp_path: Path) -> None:
    summary = run_hybrid_xiris_batch(tmp_path)

    assert summary["n_cases"] == 1000
    assert summary["accept"] == 612
    assert summary["lower_confidence"] == 201
    assert summary["block"] == 187
    assert summary["baseline_critical_misses"] == 168
    assert summary["fuzzyxai_critical_misses"] == 0
    assert summary["proof_package_valid"] is True

    saved = json.loads((tmp_path / "hybrid_xiris_batch_summary.json").read_text(encoding="utf-8"))
    assert saved == summary
