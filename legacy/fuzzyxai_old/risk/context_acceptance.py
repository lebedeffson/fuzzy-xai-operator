from __future__ import annotations

from fuzzyxai.category import ExplanationCategory, ExplanationCategoryObject, RiskContext, has_auto_accept


def observer_auto_accept_allowed(
    category: ExplanationCategory,
    obj: ExplanationCategoryObject,
    allowed_actions: set[str],
) -> bool:
    """Context-topos gate: True iff AutoAccept subcontext is inhabited."""

    return has_auto_accept(RiskContext(category, {obj: allowed_actions}), obj)
