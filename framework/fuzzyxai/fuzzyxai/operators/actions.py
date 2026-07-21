from __future__ import annotations

from collections.abc import Sequence

from fuzzyxai.core.diagnostics import DiagnosticState
from fuzzyxai.core.risk_observer import RiskResult

from .contracts import ActionResult


def select_action(risk: RiskResult, diagnostics: Sequence[DiagnosticState]) -> ActionResult:
    """Select a public action from a typed risk result and diagnostics."""

    if risk.chi_r_crit == 1:
        return ActionResult(
            action="block",
            status="blocked",
            reason="automatic acceptance is forbidden for chi_r_crit = 1",
        )
    if diagnostics:
        action = str(diagnostics[0].context.get("recommended_action", "review"))
        return ActionResult(action=action, status="warning", reason=diagnostics[0].reason)
    status = "blocked" if risk.action == "block" else "passed"
    return ActionResult(action=risk.action, status=status, reason="risk policy result")
