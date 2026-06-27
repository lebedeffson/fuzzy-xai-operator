from __future__ import annotations

from apps.services.layered_case import LayeredCaseService, build_case_state
from fuzzyxai.studio import StudioExplainPlan, WhatIfOverrides, recompute_case_state


def test_disabling_critical_rule_changes_action_from_block() -> None:
    service = LayeredCaseService.create()
    base = build_case_state(service, 'safe', sample_index=0, dataset_mode='breast_cancer')
    overrides = WhatIfOverrides(
        predicted_risk=0.20,
        uncertainty=0.10,
        i_pre=0.90,
        reduction_loss=0.04,
        chi_r=0,
        chi_r_crit=1,
        chi_auto=True,
        trace_valid=True,
    )
    strict_plan = StudioExplainPlan(use_critical_rupture=True)
    relaxed_plan = StudioExplainPlan(use_critical_rupture=False)

    strict_state = recompute_case_state(base, strict_plan, overrides)
    relaxed_state = recompute_case_state(base, relaxed_plan, overrides)

    assert strict_state['risk']['action'] == 'block'
    assert strict_state['risk']['chi_R_crit'] == 1
    assert relaxed_state['risk']['chi_R_crit'] == 0
    assert relaxed_state['risk']['action'] != 'block'
