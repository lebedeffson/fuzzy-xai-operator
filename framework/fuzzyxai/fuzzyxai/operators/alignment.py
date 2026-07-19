from __future__ import annotations

from fuzzyxai.core.alignment import AlignmentResult
from fuzzyxai.core.alignment import compute_alignment as _compute_alignment

from .contracts import AlignmentInput


def compute_alignment(request: AlignmentInput) -> AlignmentResult:
    """Compute alignment using the canonical core implementation."""

    return _compute_alignment(
        request.components,
        request.weights,
        gamma_max=request.gamma_max,
        delta_t=request.delta_t,
        delta_max=request.delta_max,
    )
