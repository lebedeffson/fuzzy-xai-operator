from __future__ import annotations

from apps.services.layered_case import LayeredCaseService, build_case_state
from fuzzyxai.studio import StudioExplainPlan, apply_named_preset


def test_critical_rupture_preset_blocks() -> None:
    service = LayeredCaseService.create()
    base = build_case_state(service, 'safe', sample_index=0, dataset_mode='breast_cancer')
    state = apply_named_preset(base, StudioExplainPlan(), 'critical_rupture')
    assert state['risk']['chi_R_crit'] == 1
    assert state['risk']['action'] == 'block'


def test_trace_gap_preset_disables_full_accept() -> None:
    service = LayeredCaseService.create()
    base = build_case_state(service, 'safe', sample_index=0, dataset_mode='breast_cancer')
    state = apply_named_preset(base, StudioExplainPlan(), 'trace_gap')
    assert state['risk']['action'] != 'accept'


def test_safe_accept_preset_sets_low_risk() -> None:
    service = LayeredCaseService.create()
    base = build_case_state(service, 'safe', sample_index=0, dataset_mode='breast_cancer')
    state = apply_named_preset(base, StudioExplainPlan(), 'safe_accept')
    assert float(state['risk']['rho']) < 0.25
