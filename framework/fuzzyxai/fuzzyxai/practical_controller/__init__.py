"""Practical budgeted action controller for operational FuzzyXAI use."""

from .calibration import apply_calibrator, compare_calibrators, fit_calibrator
from .benchmarks import (
    BASELINE_POLICIES,
    allocate_score_budget,
    compare_at_matched_budgets,
    component_ablation_scores,
    policy_metrics,
)
from .canonical import (
    CanonicalExplanation,
    CanonicalReason,
    PresentationProjection,
    PresentationReason,
    project_explanation,
    projection_metrics,
)
from .contracts import (
    ActionAssessment,
    BatchAssessment,
    CostProfile,
    CostProfileName,
    DeploymentContext,
    DeploymentMode,
    ExplanationArtifact,
    GuardResult,
    HardGuardStatus,
    PracticalDevelopmentExample,
    PracticalPolicy,
    PredictionArtifact,
    ReviewBudget,
    RouteArtifacts,
    cost_profile,
)
from .deployment import CanaryPolicy, DeploymentRecord, MonitoringSnapshot, ShadowCanaryMonitor
from .guards import evaluate_hard_guard
from .runtime import assess_action, assess_batch, assess_stream, verify_replay
from .training import fit_practical_policy

__all__ = [
    "ActionAssessment",
    "BASELINE_POLICIES",
    "BatchAssessment",
    "CanonicalExplanation",
    "CanonicalReason",
    "CanaryPolicy",
    "CostProfile",
    "CostProfileName",
    "DeploymentContext",
    "DeploymentMode",
    "DeploymentRecord",
    "ExplanationArtifact",
    "GuardResult",
    "HardGuardStatus",
    "MonitoringSnapshot",
    "PracticalDevelopmentExample",
    "PracticalPolicy",
    "PredictionArtifact",
    "PresentationProjection",
    "PresentationReason",
    "ReviewBudget",
    "RouteArtifacts",
    "ShadowCanaryMonitor",
    "apply_calibrator",
    "allocate_score_budget",
    "assess_action",
    "assess_batch",
    "assess_stream",
    "compare_calibrators",
    "compare_at_matched_budgets",
    "component_ablation_scores",
    "cost_profile",
    "evaluate_hard_guard",
    "fit_calibrator",
    "fit_practical_policy",
    "project_explanation",
    "projection_metrics",
    "policy_metrics",
    "verify_replay",
]
