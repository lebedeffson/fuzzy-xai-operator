from __future__ import annotations

from typing import Any, Mapping

import numpy as np

from .contracts import ExplanationEvidence, ExplanationGraph

_NOT_EVALUATED_REASONS = {
    "faithfulness": "no perturbation-based faithfulness check was supplied",
    "fidelity": "no surrogate fidelity measurement was supplied",
    "stability": "no repeated-perturbation stability check was supplied",
    "coverage": "no class concept exposed a measured primary_rule_coverage",
    "consistency": "no cross-explainer consistency check was supplied",
    "calibration": "calibration requires a separate held-out calibration/validation set, which was not supplied",
    "sparsity": "no numeric local contributions were available",
    "rule_complexity": "no rules were extracted for this model",
    "counterfactual_validity": "no counterfactuals were generated",
    "prototype_representativeness": "no class concept exposed a measured intra_class_variability",
    "trace_completeness": "the explanation graph has no nodes",
    "reconstruction_error": "no model family with a computable reconstruction chain (e.g. linear) was present, and none was supplied",
}

_METHODS = {
    "faithfulness": "supplied by caller (perturbation-based faithfulness check)",
    "fidelity": "supplied by caller (surrogate fidelity measurement)",
    "stability": "supplied by caller (repeated-perturbation stability check)",
    "coverage": "mean of ClassConcept.primary_rule_coverage across concepts",
    "consistency": "supplied by caller (cross-explainer consistency check)",
    "calibration": "supplied by caller (held-out calibration/validation set)",
    "sparsity": "1 / count(non-zero local contributions)",
    "rule_complexity": "mean of LearnedRule.complexity across extracted rules",
    "counterfactual_validity": "fraction of counterfactuals whose source_prediction != target_prediction",
    "prototype_representativeness": "1 / (1 + mean(ClassConcept.intra_class_variability))",
    "trace_completeness": "fraction of explanation-graph nodes with evidence_refs or a terminal node_type",
    "reconstruction_error": "mean of ModelInternalsEvidence.reconstruction_error (|reconstructed_score - actual_score|, reconstructed from transformed features, coefficients, and intercept -- a display-fidelity sanity check, not the reduction loss Delta)",
}

_SOURCE_REFS = {
    "coverage": ("concepts",),
    "sparsity": ("contributions",),
    "rule_complexity": ("rules",),
    "counterfactual_validity": ("counterfactuals",),
    "prototype_representativeness": ("concepts",),
    "trace_completeness": ("explanation_graph",),
    "reconstruction_error": ("model_internals",),
}


def _compute_quality(
    evidence: ExplanationEvidence,
    graph: ExplanationGraph,
    *,
    contributions: Mapping[str, Any] | None,
    supplied_metrics: Mapping[str, float] | None,
) -> dict[str, float | None]:
    supplied = dict(supplied_metrics or {})
    rules = list(evidence.rules)
    concepts = list(evidence.concepts)
    counterfactuals = list(evidence.counterfactuals)
    numeric_contributions = [abs(float(value)) for value in (contributions or {}).values() if isinstance(value, (int, float))]
    trace_nodes = len(graph.nodes)
    traced_nodes = sum(bool(node.evidence_refs) or node.node_type in {"prediction", "action"} for node in graph.nodes)
    rule_complexities = [rule.complexity for rule in rules]
    known_coverage = [concept.primary_rule_coverage for concept in concepts if concept.primary_rule_coverage is not None]
    valid_counterfactuals = [item.source_prediction != item.target_prediction for item in counterfactuals]
    variability = [item.intra_class_variability for item in concepts if item.intra_class_variability is not None]
    measured_reconstruction_errors = [item.reconstruction_error for item in evidence.model_internals if item.reconstruction_error is not None]
    reconstruction_error = supplied.get("reconstruction_error")
    if reconstruction_error is None and measured_reconstruction_errors:
        reconstruction_error = float(np.mean(measured_reconstruction_errors))
    return {
        "faithfulness": supplied.get("faithfulness"),
        "fidelity": supplied.get("fidelity"),
        "stability": supplied.get("stability"),
        "coverage": None if not known_coverage else float(np.mean(known_coverage)),
        "consistency": supplied.get("consistency"),
        "calibration": supplied.get("calibration"),
        "sparsity": None if not numeric_contributions else 1.0 / len(numeric_contributions),
        "rule_complexity": None if not rule_complexities else float(np.mean(rule_complexities)),
        "counterfactual_validity": None if not valid_counterfactuals else float(np.mean(valid_counterfactuals)),
        "prototype_representativeness": None if not variability else 1.0 / (1.0 + float(np.mean(variability))),
        "trace_completeness": None if trace_nodes == 0 else traced_nodes / trace_nodes,
        "reconstruction_error": reconstruction_error,
    }


def evaluate_explanation_quality(
    evidence: ExplanationEvidence,
    graph: ExplanationGraph,
    *,
    contributions: Mapping[str, Any] | None = None,
    supplied_metrics: Mapping[str, float] | None = None,
) -> dict[str, float | None]:
    """Compute only quality metrics supported by available explanation evidence."""

    return _compute_quality(evidence, graph, contributions=contributions, supplied_metrics=supplied_metrics)


def evaluate_explanation_quality_status(
    evidence: ExplanationEvidence,
    graph: ExplanationGraph,
    *,
    contributions: Mapping[str, Any] | None = None,
    supplied_metrics: Mapping[str, float] | None = None,
) -> dict[str, dict[str, Any]]:
    """P15.13/P16 section 19: a structured record for every quality metric —
    its value (or None), a status ("measured" or "not_evaluated" — this
    function does not yet know model capabilities, so it never claims
    "not_applicable"; a capability-aware caller may upgrade that), the
    computation method, a concrete reason when unmeasured, and which
    evidence layer it was drawn from.

    Kept as a separate function (rather than changing
    ``evaluate_explanation_quality``'s return shape) so the existing public,
    manifest-tracked contract stays exactly backward compatible.
    """

    metrics = _compute_quality(evidence, graph, contributions=contributions, supplied_metrics=supplied_metrics)
    return {
        name: {
            "value": value,
            "status": "measured" if value is not None else "not_evaluated",
            "method": _METHODS.get(name, ""),
            "reason": "" if value is not None else _NOT_EVALUATED_REASONS.get(name, "not measured for this explanation"),
            "source_refs": list(_SOURCE_REFS.get(name, ())) if value is not None else [],
        }
        for name, value in metrics.items()
    }
