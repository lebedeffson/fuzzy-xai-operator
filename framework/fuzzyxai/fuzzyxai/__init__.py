from .core.explanation_object import ExplanationObject, Rule, Trace
from .core.explain_plan import ExplainPlan
from .core.composition import compose
from .core.trust_evaluator import semantic_disagreement, interpretability_loss, interpretability_index
from .core.system_operator import SystemOperator
from .hierarchy.f0 import F0
from .hierarchy.interval import IntervalFS
from .hierarchy.hesitant import HesitantFS
from .hierarchy.neutrosophic import NeutrosophicFS
from .hierarchy.multilevel import MultiLevelFS
from .hierarchy.reductions import reduce_to_f0, reduction_loss
from .hierarchy.reductions import reduce_to_f0, reduction_loss
from .selection.profile_builder import build_profile
from .selection.pareto_selector import Candidate, select_minimal_sufficient, pareto_front

__all__ = [
    'ExplanationObject','Rule','Trace','ExplainPlan','compose',
    'semantic_disagreement','interpretability_loss','interpretability_index','SystemOperator',
    'F0','IntervalFS','HesitantFS','NeutrosophicFS','MultiLevelFS',
    'reduce_to_f0','reduction_loss','build_profile','Candidate','select_minimal_sufficient','pareto_front'
]
from .api import FuzzyXAIPipeline, ExplanationResult
from .risk import RiskAction, RiskDecision, RiskPolicy, RiskAwareModel, RiskAwareObserver
from .data import DatasetRecord, infer_dataset_profile
from .rules import bootstrap_lofo_f1_importance, lofo_f1_importance, select_top_rules_by_lofo_f1
from .trust import compute_interpretability_index

__all__ += ['FuzzyXAIPipeline', 'ExplanationResult']
__all__ += ['RiskAction', 'RiskDecision', 'RiskPolicy', 'RiskAwareModel', 'RiskAwareObserver']
__all__ += ['DatasetRecord', 'infer_dataset_profile']
__all__ += ['lofo_f1_importance', 'bootstrap_lofo_f1_importance', 'select_top_rules_by_lofo_f1']
__all__ += ['compute_interpretability_index']

__version__ = "1.0.0"


def build_explanation(*, terms, representation, rules=None, activations=None, uncertainty=0.0, trace=None, metadata=None):
    """Build a traceable explanation object E_k."""
    from datetime import datetime, timezone

    trace = trace or Trace(
        id="manual",
        version=__version__,
        timestamp=datetime.now(timezone.utc).isoformat(),
    )
    return ExplanationObject(
        terms=set(terms),
        representation=representation,
        rules=list(rules or []),
        activations=dict(activations or {}),
        uncertainty=float(uncertainty),
        trace=trace,
        metadata=dict(metadata or {}),
    )


def compute_alignment(components, weights, gamma_max=1.0, delta_t=0.0, delta_max=1.0):
    """Compute T_ij alignment for explanation components."""
    from .core.alignment import compute_alignment as _compute_alignment

    return _compute_alignment(components, weights, gamma_max, delta_t, delta_max)


def compute_reduction_loss(components, weights):
    """Compute reduction loss Delta."""
    from .core.reduction import compute_reduction_loss as _compute_reduction_loss

    return _compute_reduction_loss(components, weights)


def observe_risk(components, weights, thresholds, chi_r_crit=0):
    """Compute rho and select an action through the risk observer."""
    from .core.risk_observer import observe_risk as _observe_risk

    return _observe_risk(components, weights, thresholds, chi_r_crit)


def diagnose(code, reason, severity="warning", context=None):
    """Create a diagnostic state."""
    from .core.diagnostics import DiagnosticState

    return DiagnosticState(code=code, reason=reason, severity=severity, context=dict(context or {}))


def make_action(rho, chi_r_crit, thresholds):
    """Select the action for a risk value and critical-risk flag."""
    from .core.risk_observer import compute_action

    return compute_action(rho, chi_r_crit, thresholds)


def build_proof_trace(scenario, report, engine_payload=None):
    """Build a verifiable proof trace."""
    from .core.proof_package import build_proof_package

    return build_proof_package(scenario, report, engine_payload)


def verify_proof_trace(package, require_current_code_version=False):
    """Verify a FuzzyXAI proof trace."""
    from .core.proof_package import verify_proof_package

    return verify_proof_package(package, require_current_code_version=require_current_code_version)


def show_operator_route():
    """Return the invariant FuzzyXAI operator route."""
    return [
        "input",
        "adapter",
        "E_k/D_k",
        "T_ij",
        "F",
        "Delta",
        "risk_observer",
        "action",
        "proof_trace",
    ]


__all__ += [
    "__version__",
    "build_explanation",
    "compute_alignment",
    "compute_reduction_loss",
    "observe_risk",
    "diagnose",
    "make_action",
    "build_proof_trace",
    "verify_proof_trace",
    "show_operator_route",
]

# v0.3 framework-core public API.
from .core.explanation import build_explainable_object
from .core.route import build_route
from .proof.trace import build_proof_trace
from .proof.verifier import verify_proof_trace
from .viz.export import save_route_json
from .viz.matplotlib_dashboard import render_dashboard
from .examples import list_examples, load_example
from .runtime import FuzzyXAI
from .adapters import BaseAdapter, TabularClassificationAdapter, get_adapter, list_adapters
from .operators import get_operator, list_operators
from .visualization import (
    render_action_boundary,
    render_coverage_curve,
    render_gamma_delta_action_map,
    render_operator_trace_heatmap,
    render_proof_consistency_matrix,
    render_representation_atlas,
    render_risk_waterfall,
    render_route_sankey,
)

__all__ += [
    "FuzzyXAI",
    "build_explainable_object",
    "build_route",
    "build_proof_trace",
    "verify_proof_trace",
    "render_dashboard",
    "save_route_json",
    "list_examples",
    "load_example",
    "BaseAdapter",
    "TabularClassificationAdapter",
    "get_adapter",
    "list_adapters",
    "get_operator",
    "list_operators",
    "render_route_sankey",
    "render_gamma_delta_action_map",
    "render_risk_waterfall",
    "render_operator_trace_heatmap",
    "render_representation_atlas",
    "render_coverage_curve",
    "render_action_boundary",
    "render_proof_consistency_matrix",
]
