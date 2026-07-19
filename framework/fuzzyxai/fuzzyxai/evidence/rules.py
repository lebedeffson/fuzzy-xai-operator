from __future__ import annotations

from dataclasses import replace
from typing import Any, Mapping, Sequence

import numpy as np

from fuzzyxai.adapters.model import ModelAdapter

from .contracts import LearnedRule


def _score_rule(rule: LearnedRule) -> float:
    components = [
        value
        for value in (
            rule.coverage,
            rule.precision,
            rule.stability,
            abs(rule.counterfactual_effect.get("validation", 0.0)) if rule.counterfactual_effect else None,
        )
        if value is not None
    ]
    if not components:
        return 0.0
    return float(sum(components) / len(components) / max(1.0, 0.25 * rule.complexity))


def rank_rules(rules: Sequence[LearnedRule], *, primary_limit: int = 7) -> list[LearnedRule]:
    """Rank rules using available evidence without replacing missing components."""

    scored = [(rule.importance if rule.importance is not None else _score_rule(rule), rule) for rule in rules]
    scored.sort(key=lambda item: (-item[0], item[1].complexity, item[1].rule_id))
    return [replace(rule, importance=round(score, 6), is_primary=index < primary_limit) for index, (score, rule) in enumerate(scored)]


def rule_complexity(rule: LearnedRule) -> dict[str, float | int | bool]:
    """Return cognitive and structural complexity of one rule."""

    return {
        "condition_count": len(rule.antecedents),
        "complexity": float(rule.complexity),
        "is_redundant": rule.is_redundant,
        "is_conflicting": rule.is_conflicting,
        "cognitive_load": float(rule.complexity + int(rule.is_conflicting) + 0.5 * int(rule.is_redundant)),
    }


def _tree_rules(
    estimator: Any,
    feature_names: Sequence[str],
    *,
    model_version: str,
    prefix: str,
    max_rules: int,
) -> list[LearnedRule]:
    tree = estimator.tree_
    classes = list(getattr(estimator, "classes_", []))
    results: list[LearnedRule] = []
    root_support = max(int(tree.n_node_samples[0]), 1)

    def walk(node: int, conditions: list[str]) -> None:
        if len(results) >= max_rules:
            return
        left = int(tree.children_left[node])
        right = int(tree.children_right[node])
        if left == right:
            values = np.asarray(tree.value[node]).reshape(-1)
            total = float(values.sum())
            distribution = {
                str(classes[index] if index < len(classes) else index): float(value / total)
                for index, value in enumerate(values)
                if total > 0
            }
            winner = int(np.argmax(values)) if len(values) else 0
            consequent = str(classes[winner] if winner < len(classes) else winner)
            precision = float(values[winner] / total) if total else None
            support = int(tree.n_node_samples[node])
            rule_id = f"{prefix}_{len(results):03d}"
            results.append(
                LearnedRule(
                    rule_id=rule_id,
                    model_version=model_version,
                    antecedents=list(conditions),
                    consequent=consequent,
                    activation=None,
                    coverage=support / root_support,
                    precision=precision,
                    support=support,
                    stability=None,
                    importance=None,
                    counterfactual_effect={},
                    source_objects=[],
                    class_distribution=distribution,
                    human_text=(" and ".join(conditions) if conditions else "all objects") + f" -> class {consequent}",
                    complexity=float(len(conditions)),
                    is_primary=False,
                    is_redundant=False,
                    is_conflicting=False,
                    native=True,
                    surrogate=False,
                    evidence_refs=[f"model.tree_.node:{node}"],
                )
            )
            return
        feature_index = int(tree.feature[node])
        feature = feature_names[feature_index] if feature_index < len(feature_names) else f"feature_{feature_index}"
        threshold = float(tree.threshold[node])
        walk(left, [*conditions, f"{feature} <= {threshold:.6g}"])
        walk(right, [*conditions, f"{feature} > {threshold:.6g}"])

    walk(0, [])
    return results


def _native_rules(adapter: ModelAdapter, model_version: str) -> list[LearnedRule]:
    raw_rules = adapter.extract_rules()
    results: list[LearnedRule] = []
    for index, raw in enumerate(raw_rules):
        antecedents = raw.get("antecedents", raw.get("conditions", []))
        if isinstance(antecedents, Mapping):
            antecedents = [f"{key} is {value}" for key, value in antecedents.items()]
        consequent = str(raw.get("consequent", raw.get("class", "unknown")))
        results.append(
            LearnedRule(
                rule_id=str(raw.get("rule_id", f"native_{index:03d}")),
                model_version=model_version,
                antecedents=[str(item) for item in antecedents],
                consequent=consequent,
                activation=_optional_float(raw.get("activation")),
                coverage=_optional_float(raw.get("coverage")),
                precision=_optional_float(raw.get("precision")),
                support=_optional_int(raw.get("support")),
                stability=_optional_float(raw.get("stability")),
                importance=_optional_float(raw.get("importance")),
                counterfactual_effect=dict(raw.get("counterfactual_effect", {})),
                source_objects=[str(item) for item in raw.get("source_objects", [])],
                class_distribution={str(key): float(value) for key, value in raw.get("class_distribution", {}).items()},
                human_text=str(raw.get("human_text", f"{' and '.join(antecedents)} -> {consequent}")),
                complexity=float(raw.get("complexity", len(antecedents))),
                is_primary=bool(raw.get("is_primary", False)),
                is_redundant=bool(raw.get("is_redundant", False)),
                is_conflicting=bool(raw.get("is_conflicting", False)),
                native=True,
                surrogate=False,
                evidence_refs=[str(item) for item in raw.get("evidence_refs", ["adapter.extract_rules"])],
            )
        )
    return results


def _linear_rules(adapter: ModelAdapter, feature_names: Sequence[str], model_version: str) -> list[LearnedRule]:
    coefficients = np.asarray(getattr(adapter.model, "coef_", []), dtype=float)
    if coefficients.ndim == 1:
        coefficients = coefficients.reshape(1, -1)
    classes = list(getattr(adapter.model, "classes_", range(coefficients.shape[0])))
    results: list[LearnedRule] = []
    for class_index, row in enumerate(coefficients):
        consequent = str(classes[class_index] if class_index < len(classes) else class_index)
        for feature_index in np.argsort(np.abs(row))[::-1]:
            coefficient = float(row[feature_index])
            if abs(coefficient) <= 1e-12:
                continue
            feature = feature_names[feature_index] if feature_index < len(feature_names) else f"feature_{feature_index}"
            direction = "higher" if coefficient > 0 else "lower"
            results.append(
                LearnedRule(
                    rule_id=f"linear_{class_index}_{feature_index}",
                    model_version=model_version,
                    antecedents=[f"{feature} is {direction}"],
                    consequent=consequent,
                    activation=None,
                    coverage=None,
                    precision=None,
                    support=None,
                    stability=None,
                    importance=abs(coefficient),
                    counterfactual_effect={},
                    source_objects=[],
                    class_distribution={},
                    human_text=f"{feature} has a {direction} linear contribution to class {consequent}",
                    complexity=1.0,
                    is_primary=False,
                    is_redundant=False,
                    is_conflicting=False,
                    native=False,
                    surrogate=True,
                    fidelity=1.0,
                    evidence_refs=[f"model.coef_[{class_index},{feature_index}]"],
                )
            )
    return results


def extract_rules(
    adapter: ModelAdapter,
    *,
    feature_names: Sequence[str] | None = None,
    model_version: str = "unknown",
    max_rules: int = 50,
    primary_limit: int = 7,
) -> list[LearnedRule]:
    """Extract native rules or explicitly labelled rule-like evidence."""

    names = list(feature_names or adapter.feature_names())
    model = adapter.model
    rules: list[LearnedRule] = []
    if adapter.capabilities().get("rules", False):
        rules.extend(_native_rules(adapter, model_version))
    elif hasattr(model, "tree_"):
        rules.extend(_tree_rules(model, names, model_version=model_version, prefix="tree", max_rules=max_rules))
    elif hasattr(model, "estimators_"):
        estimators = np.asarray(model.estimators_, dtype=object).reshape(-1)
        for estimator_index, estimator in enumerate(estimators):
            if not hasattr(estimator, "tree_"):
                continue
            remaining = max_rules - len(rules)
            if remaining <= 0:
                break
            rules.extend(
                _tree_rules(
                    estimator,
                    names,
                    model_version=model_version,
                    prefix=f"tree_{estimator_index}",
                    max_rules=remaining,
                )
            )
    elif hasattr(model, "coef_"):
        rules.extend(_linear_rules(adapter, names, model_version))
    return rank_rules(rules[:max_rules], primary_limit=primary_limit)


def evaluate_rule_ablation(
    rule: LearnedRule,
    *,
    baseline_metrics: Mapping[str, float],
    ablated_metrics: Mapping[str, float],
) -> LearnedRule:
    """Attach measured train/validation/test effects of disabling a rule."""

    shared = sorted(set(baseline_metrics) & set(ablated_metrics))
    if not shared:
        raise ValueError("baseline_metrics and ablated_metrics require at least one shared metric")
    effects = {name: float(baseline_metrics[name] - ablated_metrics[name]) for name in shared}
    importance = effects.get("validation", effects.get("test", sum(effects.values()) / len(effects)))
    return replace(
        rule,
        counterfactual_effect={name: round(value, 6) for name, value in effects.items()},
        importance=round(float(importance), 6),
        evidence_refs=[*rule.evidence_refs, "measured_rule_ablation"],
    )


def _optional_float(value: Any) -> float | None:
    return None if value is None else float(value)


def _optional_int(value: Any) -> int | None:
    return None if value is None else int(value)
