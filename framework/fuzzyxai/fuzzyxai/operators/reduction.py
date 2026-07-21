from __future__ import annotations

from fuzzyxai.core.reduction import compute_reduction as _compute_reduction

from .contracts import ReductionInput, ReductionOperatorResult


def compute_reduction(request: ReductionInput) -> ReductionOperatorResult:
    """Compute reduction loss without scenario-specific default values."""

    result = _compute_reduction(request.components, request.weights, request.delta_max)
    return ReductionOperatorResult(
        delta=result.delta,
        delta_max=result.delta_max,
        r_delta=round(min(1.0, float(request.kappa_delta) * result.delta), 6),
        allowed=result.allowed,
    )
