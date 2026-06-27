import pytest

from fuzzyxai.core.alignment import compute_gamma
from fuzzyxai.core.reduction import compute_reduction_loss
from fuzzyxai.core.risk_observer import compute_action, compute_risk
from fuzzyxai.core.scenario_engine import DEFAULT_HYBRID_PLAN, compute_hybrid_xiris


def test_hybrid_gamma_components() -> None:
    assert compute_gamma(DEFAULT_HYBRID_PLAN.gamma_components, DEFAULT_HYBRID_PLAN.gamma_weights) == pytest.approx(0.351)


def test_hybrid_reduction_loss() -> None:
    assert compute_reduction_loss(DEFAULT_HYBRID_PLAN.reduction_components, DEFAULT_HYBRID_PLAN.reduction_weights) == pytest.approx(0.106811)


def test_hybrid_reduction_component() -> None:
    result = compute_hybrid_xiris()
    risk = next(item for item in result.operator_values if item["node_id"] == "risk_observer")
    assert risk["computed"]["rho"] == pytest.approx(0.800)


def test_hybrid_risk_decomposition() -> None:
    assert compute_risk(
        {"model_signal": 0.88, "block_rule": 0.91, "source_conflict": 1.0, "reduction_component": 0.3225},
        DEFAULT_HYBRID_PLAN.risk_weights,
    ) == pytest.approx(0.800)


def test_hybrid_action_rule() -> None:
    assert compute_action(0.1, chi_r_crit=1, thresholds=DEFAULT_HYBRID_PLAN.thresholds) == "block"


def test_hybrid_diagnostic_type() -> None:
    result = compute_hybrid_xiris()
    assert result.diagnostic_id == "D_quality_source_conflict"
    assert result.diagnostic_type == "quality_source_conflict"
