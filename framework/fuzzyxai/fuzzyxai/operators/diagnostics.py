from __future__ import annotations

from collections.abc import Mapping, Sequence

from fuzzyxai.core.diagnostics import DiagnosticState

from .contracts import DiagnosticRule


def diagnose_route(conditions: Mapping[str, bool], rules: Sequence[DiagnosticRule]) -> list[DiagnosticState]:
    """Evaluate declarative diagnostic rules for any model route."""

    diagnostics: list[DiagnosticState] = []
    for rule in rules:
        if conditions.get(rule.condition, False):
            context = dict(rule.context)
            context["condition"] = rule.condition
            context["recommended_action"] = rule.recommended_action
            diagnostics.append(
                DiagnosticState(
                    code=rule.code,
                    reason=rule.reason,
                    severity=rule.severity,
                    context=context,
                )
            )
    return diagnostics
