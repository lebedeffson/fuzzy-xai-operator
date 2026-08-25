from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

from .contracts import FuzzyRuleActivation, FuzzyTermMembership


def collect_fuzzy_rule_activations(
    raw_activations: Sequence[Mapping[str, Any]],
    *,
    object_id: str,
) -> list[FuzzyRuleActivation]:
    """Convert an adapter's raw ``activated_rules`` channel into typed evidence.

    Any rule/fuzzy model adapter can populate this channel with plain dicts
    shaped like::

        {
            "rule_id": "R7",
            "terms": [{"feature": "A", "term": "high", "membership_degree": 0.9}, ...],
            "activation_strength": 0.82,
            "conclusion": "1",
            "feature_values": {"A": 23.5},  # optional
        }

    This is not tied to any specific ANFIS library — it's a generic
    contract. A malformed entry (missing required fields, an out-of-range
    membership degree) is dropped with an explicit, traceable limitation
    rather than silently coerced into something that looks valid.
    """

    results: list[FuzzyRuleActivation] = []
    for index, raw in enumerate(raw_activations):
        rule_id = str(raw.get("rule_id", "")).strip()
        raw_terms = raw.get("terms")
        activation_strength = raw.get("activation_strength")
        conclusion = raw.get("conclusion")
        feature_values = raw.get("feature_values") if isinstance(raw.get("feature_values"), Mapping) else {}
        limitations: list[str] = []
        if not rule_id:
            limitations.append(f"activated_rules[{index}] has no rule_id; dropped")
            continue
        if not isinstance(raw_terms, Sequence) or not raw_terms:
            limitations.append(f"rule {rule_id} has no antecedent terms; dropped")
            continue
        if not isinstance(activation_strength, (int, float)) or not 0.0 <= float(activation_strength) <= 1.0:
            limitations.append(f"rule {rule_id} activation_strength is missing or out of [0, 1]; dropped")
            continue
        if conclusion is None or not str(conclusion).strip():
            limitations.append(f"rule {rule_id} has no conclusion; dropped")
            continue
        terms: list[FuzzyTermMembership] = []
        dropped_terms: list[str] = []
        for term in raw_terms:
            if not isinstance(term, Mapping):
                dropped_terms.append(str(term))
                continue
            feature = str(term.get("feature", "")).strip()
            label = str(term.get("term", "")).strip()
            degree = term.get("membership_degree")
            if not feature or not label or not isinstance(degree, (int, float)) or not 0.0 <= float(degree) <= 1.0:
                dropped_terms.append(str(term))
                continue
            value = term.get("feature_value", feature_values.get(feature))
            terms.append(
                FuzzyTermMembership(
                    feature=feature,
                    term=label,
                    membership_degree=float(degree),
                    feature_value=float(value) if isinstance(value, (int, float)) else None,
                )
            )
        if dropped_terms:
            limitations.append(f"rule {rule_id}: {len(dropped_terms)} malformed term(s) dropped")
        if not terms:
            limitations.append(f"rule {rule_id} had only malformed terms; dropped entirely")
            continue
        results.append(
            FuzzyRuleActivation(
                object_id=str(object_id),
                rule_id=rule_id,
                terms=tuple(terms),
                activation_strength=float(activation_strength),
                conclusion=str(conclusion),
                limitations=tuple(limitations),
            )
        )
    return results
