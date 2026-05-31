from __future__ import annotations

import pandas as pd

from fuzzyxai.datasets import load_dataset_mode
from fuzzyxai.risk.decision_policy import RiskAction, RiskPolicy


def test_synthetic_rupture_dataset_has_expected_fields() -> None:
    _record, df = load_dataset_mode('synthetic_ruptures')
    for col in ('chi_R', 'chi_R_crit', 'critical_rupture', 'rupture_type', 'expected_action'):
        assert col in df.columns
    crit = df[df['chi_R_crit'].astype(int) == 1]
    assert not crit.empty
    assert (crit['expected_action'].astype(str) == 'block').all()


def test_critical_rupture_always_blocks() -> None:
    policy = RiskPolicy()
    decision = policy.choose_from_risk(
        application_risk=0.05,
        uncertainty=0.01,
        predicted_risk=0.05,
        pre_interpretability=0.95,
        reduction_loss=0.01,
        diagnostics=[],
        chi_r=1,
        chi_r_crit=1,
        chi_auto=True,
        trace_valid=True,
    )
    assert decision.action == RiskAction.BLOCK


def test_noncritical_rupture_not_auto_accept() -> None:
    policy = RiskPolicy()
    decision = policy.choose_from_risk(
        application_risk=0.08,
        uncertainty=0.10,
        predicted_risk=0.10,
        pre_interpretability=0.90,
        reduction_loss=0.02,
        diagnostics=[],
        chi_r=1,
        chi_r_crit=0,
        chi_auto=True,
        trace_valid=True,
    )
    assert decision.action != RiskAction.ACCEPT


def test_clean_case_can_accept() -> None:
    policy = RiskPolicy()
    decision = policy.choose_from_risk(
        application_risk=0.05,
        uncertainty=0.05,
        predicted_risk=0.08,
        pre_interpretability=0.90,
        reduction_loss=0.03,
        diagnostics=[],
        chi_r=0,
        chi_r_crit=0,
        chi_auto=True,
        trace_valid=True,
    )
    assert decision.action == RiskAction.ACCEPT


def test_critical_aggregation_preserves_any_edge() -> None:
    edges = pd.DataFrame({'critical_edge': [0, 1, 0]})
    chi_r_crit = int(edges['critical_edge'].astype(bool).any())
    assert chi_r_crit == 1

