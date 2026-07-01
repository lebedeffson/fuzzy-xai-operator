from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .state import StudioExplainPlan, WhatIfOverrides, recompute_case_state


@dataclass(frozen=True)
class ScenarioPreset:
    name: str
    description: str
    overrides: WhatIfOverrides


PRESETS: dict[str, ScenarioPreset] = {
    'safe_accept': ScenarioPreset(
        name='safe_accept',
        description='Low risk, no rupture, context allows auto.',
        overrides=WhatIfOverrides(predicted_risk=0.12, uncertainty=0.10, i_pre=0.88, reduction_loss=0.04, chi_r=0, chi_r_crit=0, chi_auto=True, trace_valid=True),
    ),
    'high_uncertainty': ScenarioPreset(
        name='high_uncertainty',
        description='High uncertainty drives request_more_data.',
        overrides=WhatIfOverrides(predicted_risk=0.46, uncertainty=0.62, i_pre=0.66, reduction_loss=0.10, chi_r=0, chi_r_crit=0, chi_auto=True, trace_valid=True),
    ),
    'need_more_data': ScenarioPreset(
        name='need_more_data',
        description='Trace gap and weak interpretability.',
        overrides=WhatIfOverrides(predicted_risk=0.44, uncertainty=0.40, i_pre=0.52, reduction_loss=0.11, chi_r=0, chi_r_crit=0, chi_auto=True, trace_valid=False),
    ),
    'context_forbidden': ScenarioPreset(
        name='context_forbidden',
        description='Topos context forbids automatic action.',
        overrides=WhatIfOverrides(predicted_risk=0.55, uncertainty=0.28, i_pre=0.74, reduction_loss=0.08, chi_r=0, chi_r_crit=0, chi_auto=False, trace_valid=True),
    ),
    'critical_rupture': ScenarioPreset(
        name='critical_rupture',
        description='Critical rupture must force block.',
        overrides=WhatIfOverrides(predicted_risk=0.70, uncertainty=0.50, i_pre=0.62, reduction_loss=0.12, chi_r=1, chi_r_crit=1, chi_auto=False, trace_valid=True),
    ),
    'source_conflict': ScenarioPreset(
        name='source_conflict',
        description='Source conflict triggers rupture-aware path.',
        overrides=WhatIfOverrides(predicted_risk=0.58, uncertainty=0.38, i_pre=0.68, reduction_loss=0.10, chi_r=1, chi_r_crit=0, chi_auto=True, trace_valid=True, source_conflict=True),
    ),
    'trace_gap': ScenarioPreset(
        name='trace_gap',
        description='Missing trace fields block full auto-accept.',
        overrides=WhatIfOverrides(predicted_risk=0.36, uncertainty=0.35, i_pre=0.73, reduction_loss=0.09, chi_r=0, chi_r_crit=0, chi_auto=True, trace_valid=False),
    ),
    'reduction_loss_too_high': ScenarioPreset(
        name='reduction_loss_too_high',
        description='High Delta makes observer conservative.',
        overrides=WhatIfOverrides(predicted_risk=0.42, uncertainty=0.26, i_pre=0.75, reduction_loss=0.36, chi_r=0, chi_r_crit=0, chi_auto=True, trace_valid=True),
    ),
}


def list_presets() -> list[ScenarioPreset]:
    return list(PRESETS.values())


def apply_named_preset(case_state: dict[str, Any], plan: StudioExplainPlan, preset_name: str) -> dict[str, Any]:
    preset = PRESETS.get(str(preset_name))
    if preset is None:
        return recompute_case_state(case_state, plan)
    return recompute_case_state(case_state, plan, preset.overrides)
