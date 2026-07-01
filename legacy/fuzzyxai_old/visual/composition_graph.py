from __future__ import annotations

from typing import Iterable, Mapping, Sequence, Tuple

from fuzzyxai.core.explanation_object import ExplanationObject
from fuzzyxai.core.trust_evaluator import semantic_disagreement


def _severity(gamma: float) -> str:
    if gamma < 0.20:
        return "green"
    if gamma < 0.45:
        return "orange"
    return "red"


def edge_report(edges: Sequence[Tuple[str, ExplanationObject, str, ExplanationObject]], beta: Mapping[str, float]):
    """Return a list of diagnostic edge records for a compositional pipeline."""
    rows = []
    for left_name, left, right_name, right in edges:
        gamma = semantic_disagreement(left, right, beta)
        rows.append({
            "source": left_name,
            "target": right_name,
            "gamma": round(float(gamma), 6),
            "severity": _severity(float(gamma)),
            "left_terms": sorted(left.terms),
            "right_terms": sorted(right.terms),
            "left_active_rules": sorted(left.active_rules),
            "right_active_rules": sorted(right.active_rules),
            "left_delta": left.reduction_loss,
            "right_delta": right.reduction_loss,
        })
    return rows


def composition_graph_dot(edges: Sequence[Tuple[str, ExplanationObject, str, ExplanationObject]], beta: Mapping[str, float]) -> str:
    """Build a Graphviz DOT representation of the explanation-composition graph."""
    lines = ["digraph ExplanationComposition {", "  rankdir=LR;"]
    for row in edge_report(edges, beta):
        label = f"gamma={row['gamma']}\\n{row['severity']}"
        color = row["severity"]
        lines.append(f"  {row['source']} -> {row['target']} [label=\"{label}\", color={color}, penwidth=2];")
    lines.append("}")
    return "\n".join(lines) + "\n"
