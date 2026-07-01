from __future__ import annotations

from typing import Any

from apps.services.layered_case import LayeredCaseService, build_case_state

from .state import StudioExplainPlan, WhatIfOverrides, recompute_case_state


def build_studio_case(
    service: LayeredCaseService,
    *,
    dataset_mode: str,
    scenario: str,
    sample_index: int | None,
    plan: StudioExplainPlan,
    overrides: WhatIfOverrides | None = None,
) -> dict[str, Any]:
    base = build_case_state(service, scenario, sample_index=sample_index, dataset_mode=dataset_mode)
    return recompute_case_state(base, plan, overrides)
