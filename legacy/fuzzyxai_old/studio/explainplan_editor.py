from __future__ import annotations

from dataclasses import fields
from typing import Any

from .state import StudioExplainPlan


def plan_from_dict(data: dict[str, Any] | None = None) -> StudioExplainPlan:
    plan = StudioExplainPlan()
    if not data:
        return plan
    for f in fields(plan):
        if f.name in data:
            setattr(plan, f.name, data[f.name])
    return plan


def update_plan(plan: StudioExplainPlan, updates: dict[str, Any]) -> StudioExplainPlan:
    for f in fields(plan):
        if f.name in updates:
            setattr(plan, f.name, updates[f.name])
    return plan
