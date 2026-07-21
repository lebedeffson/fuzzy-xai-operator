"""Formative components for the strong confirmatory research cycle."""

from .grid import compare_grid_configurations
from .route import FAULT_TYPES, evaluate_route_guardrails
from .scaling import run_streaming_scalability
from .stability import attribution_stability, compare_stability
from .statistics import holm_adjust, paired_bootstrap_difference, paired_permutation_pvalue

__all__ = [
    "FAULT_TYPES",
    "attribution_stability",
    "compare_grid_configurations",
    "compare_stability",
    "evaluate_route_guardrails",
    "holm_adjust",
    "paired_bootstrap_difference",
    "paired_permutation_pvalue",
    "run_streaming_scalability",
]
