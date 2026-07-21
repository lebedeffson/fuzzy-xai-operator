from __future__ import annotations

from typing import Any, Sequence

import numpy as np

from .contracts import ClassConcept, LearnedRule


def build_class_concepts(
    values: Sequence[Sequence[float]],
    labels: Sequence[Any],
    *,
    feature_names: Sequence[str] | None = None,
    object_ids: Sequence[str] | None = None,
    rules: Sequence[LearnedRule] = (),
    representative_limit: int = 3,
) -> list[ClassConcept]:
    """Describe each class through prototypes, representative objects and rules."""

    matrix = np.asarray(values, dtype=float)
    y = np.asarray(labels)
    if matrix.ndim != 2 or len(matrix) != len(y):
        raise ValueError("values must be 2D and aligned with labels")
    names = list(feature_names or [f"feature_{index}" for index in range(matrix.shape[1])])
    ids = list(object_ids or [f"object_{index}" for index in range(len(matrix))])
    concepts: list[ClassConcept] = []
    for class_value in sorted(set(y.tolist()), key=str):
        class_indices = np.flatnonzero(y == class_value)
        other_indices = np.flatnonzero(y != class_value)
        class_values = matrix[class_indices]
        prototype = np.nanmedian(class_values, axis=0)
        distances = np.linalg.norm(np.nan_to_num(class_values - prototype), axis=1)
        representatives = [ids[class_indices[index]] for index in np.argsort(distances)[:representative_limit]]
        boundary = [ids[class_indices[index]] for index in np.argsort(distances)[-representative_limit:]]
        counterexamples: list[str] = []
        if len(other_indices):
            other_distances = np.linalg.norm(np.nan_to_num(matrix[other_indices] - prototype), axis=1)
            counterexamples = [ids[other_indices[index]] for index in np.argsort(other_distances)[:representative_limit]]
        class_rules = [rule for rule in rules if rule.consequent == str(class_value)]
        primary = [rule for rule in class_rules if rule.is_primary][:7]
        covered_ids = {object_id for rule in primary for object_id in rule.source_objects}
        known_coverage = [rule.coverage for rule in primary if rule.coverage is not None]
        if covered_ids:
            coverage = len(covered_ids & {ids[index] for index in class_indices}) / max(len(class_indices), 1)
        else:
            coverage = max(known_coverage) if known_coverage else None
        limitations: list[str] = []
        if coverage is None:
            limitations.append("primary-rule coverage is unavailable")
        elif len(known_coverage) > 1 and not covered_ids:
            limitations.append("coverage is a lower bound because overlap between rules is unavailable")
        elif coverage < 0.5:
            limitations.append("primary rules cover less than half of the class")
        rule_text = "; ".join(rule.human_text for rule in primary[:3]) or "no supported primary rules"
        coverage_text = "coverage unavailable" if coverage is None else f"supported primary-rule coverage {coverage:.1%}"
        concepts.append(
            ClassConcept(
                class_id=str(class_value),
                class_name=str(class_value),
                prototype_features={name: float(value) for name, value in zip(names, prototype)},
                prototype_embedding=[],
                primary_rules=[rule.rule_id for rule in primary],
                representative_objects=representatives,
                boundary_objects=boundary,
                counterexamples=counterexamples,
                intra_class_variability=float(np.mean(distances)) if len(distances) else None,
                human_description=f"Class {class_value}: {rule_text}; {coverage_text}.",
                primary_rule_coverage=coverage,
                uncovered_fraction=None if coverage is None else max(0.0, 1.0 - coverage),
                limitations=limitations,
            )
        )
    return concepts
