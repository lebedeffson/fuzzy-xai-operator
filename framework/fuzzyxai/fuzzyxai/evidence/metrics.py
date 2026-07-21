from __future__ import annotations

from typing import Any, Mapping

import numpy as np

from .contracts import ExplanationEvidence, ExplanationGraph


def evaluate_explanation_quality(
    evidence: ExplanationEvidence,
    graph: ExplanationGraph,
    *,
    contributions: Mapping[str, Any] | None = None,
    supplied_metrics: Mapping[str, float] | None = None,
) -> dict[str, float | None]:
    """Compute only quality metrics supported by available explanation evidence."""

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
        "reconstruction_error": supplied.get("reconstruction_error"),
    }
