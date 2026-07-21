"""Repeated leaf-rule ablation with a validation-matched random baseline."""

from __future__ import annotations

from dataclasses import asdict
from typing import Sequence

import numpy as np

from .rule_ablation import AblationPair, RuleDescriptor, select_matched_random_rule, summarize_ablation


def run_repeated_leaf_rule_ablation(
    values: np.ndarray,
    labels: np.ndarray,
    rare_mask: np.ndarray,
    *,
    folds: int = 10,
    seeds: Sequence[int] = (4201, 4202, 4203, 4204, 4205),
) -> dict[str, object]:
    from sklearn.metrics import recall_score
    from sklearn.model_selection import StratifiedKFold, train_test_split
    from sklearn.tree import DecisionTreeClassifier

    matrix = np.asarray(values, dtype=float)
    target = np.asarray(labels, dtype=int)
    rare = np.asarray(rare_mask, dtype=bool)
    pairs: list[AblationPair] = []
    descriptors: list[dict[str, object]] = []
    for seed in seeds:
        splitter = StratifiedKFold(n_splits=folds, shuffle=True, random_state=seed)
        for fold, (development, test) in enumerate(splitter.split(matrix, target)):
            train, validation = train_test_split(
                development,
                test_size=0.25,
                stratify=target[development],
                random_state=seed + fold,
            )
            model = DecisionTreeClassifier(max_depth=5, min_samples_leaf=8, random_state=seed)
            model.fit(matrix[train], target[train])
            validation_leaves = model.apply(matrix[validation])
            test_leaves = model.apply(matrix[test])
            leaf_ids = sorted(set(int(item) for item in validation_leaves))
            candidates: list[RuleDescriptor] = []
            for leaf_id in leaf_ids:
                active = validation_leaves == leaf_id
                if int(active.sum()) < 3:
                    continue
                subgroup_hits = int(np.sum(active & rare[validation]))
                subgroup_total = max(1, int(np.sum(rare[validation])))
                majority = int(np.mean(target[validation][active]) >= 0.5)
                candidates.append(
                    RuleDescriptor(
                        rule_id=f"leaf:{leaf_id}",
                        coverage=float(np.mean(active)),
                        subgroup_coverage=subgroup_hits / subgroup_total,
                        exclusivity=subgroup_hits / int(active.sum()),
                        redundancy=_leaf_redundancy(active, validation_leaves),
                        activation_frequency=float(np.mean(active)),
                        depth=_node_depth(model.tree_, leaf_id),
                        confidence=float(np.mean(target[validation][active] == majority)),
                        output_class=str(majority),
                    )
                )
            eligible = [item for item in candidates if any(other.output_class == item.output_class and other.rule_id != item.rule_id for other in candidates)]
            if not eligible:
                raise RuntimeError(f"no matchable leaf rules for seed={seed} fold={fold}")
            selected = max(eligible, key=lambda item: (item.subgroup_coverage * item.exclusivity, -item.redundancy, item.rule_id))
            matched = select_matched_random_rule(selected, candidates)
            predictions = np.asarray(model.predict(matrix[test]), dtype=int)
            selected_delta = _subgroup_recall_loss(
                predictions,
                target[test],
                rare[test],
                test_leaves == int(selected.rule_id.split(":", 1)[1]),
                recall_score,
            )
            matched_delta = _subgroup_recall_loss(
                predictions,
                target[test],
                rare[test],
                test_leaves == int(matched.rule_id.split(":", 1)[1]),
                recall_score,
            )
            pairs.append(
                AblationPair(
                    fold=fold,
                    seed=seed,
                    selected_rule=selected.rule_id,
                    matched_rule=matched.rule_id,
                    selected_delta=selected_delta,
                    matched_delta=matched_delta,
                    subgroup="predefined_rare_interaction",
                )
            )
            descriptors.append({"fold": fold, "seed": seed, "selected": asdict(selected), "matched": asdict(matched)})
    return {
        "schema_version": "1.0",
        "protocol": "validation-selected leaf rule versus coverage/depth/class matched rule",
        "folds": folds,
        "seeds": list(seeds),
        "pairs": [asdict(item) | {"specific_effect": item.specific_effect} for item in pairs],
        "descriptors": descriptors,
        "summary": summarize_ablation(pairs),
        "conditional_model": _fit_conditional_model(pairs, descriptors),
        "selection_partition": "validation",
        "effect_partition": "held_out_fold",
    }


def _fit_conditional_model(pairs: Sequence[AblationPair], descriptors: Sequence[dict[str, object]]) -> dict[str, object]:
    from sklearn.linear_model import LinearRegression
    from sklearn.metrics import mean_absolute_error, r2_score
    from sklearn.model_selection import train_test_split

    features = []
    targets = []
    for pair, row in zip(pairs, descriptors):
        selected = row["selected"]
        features.append(
            [
                float(selected["coverage"]),
                float(selected["subgroup_coverage"]),
                float(selected["exclusivity"]),
                float(selected["redundancy"]),
                float(selected["depth"]),
                float(selected["confidence"]),
            ]
        )
        targets.append(pair.specific_effect)
    train, holdout = train_test_split(np.arange(len(features)), test_size=0.2, random_state=4201)
    matrix = np.asarray(features, dtype=float)
    target = np.asarray(targets, dtype=float)
    model = LinearRegression().fit(matrix[train], target[train])
    prediction = model.predict(matrix[holdout])
    return {
        "status": "measured_exploratory_controlled",
        "fit_pairs": len(train),
        "holdout_pairs": len(holdout),
        "features": ["coverage", "subgroup_coverage", "exclusivity", "redundancy", "depth", "confidence"],
        "holdout_mae": float(mean_absolute_error(target[holdout], prediction)),
        "holdout_r2": float(r2_score(target[holdout], prediction)),
        "claim_scope": "exploratory controlled predictor; independent real-benchmark confirmation required",
    }


def _subgroup_recall_loss(
    baseline: np.ndarray,
    labels: np.ndarray,
    subgroup: np.ndarray,
    ablated_rule: np.ndarray,
    recall_score: object,
) -> float:
    if not callable(recall_score):
        raise TypeError("recall_score must be callable")
    mask = np.asarray(subgroup, dtype=bool)
    if not mask.any():
        return 0.0
    ablated = baseline.copy()
    ablated[ablated_rule] = 1 - ablated[ablated_rule]
    before = float(recall_score(labels[mask], baseline[mask], zero_division=0))
    after = float(recall_score(labels[mask], ablated[mask], zero_division=0))
    return before - after


def _leaf_redundancy(active: np.ndarray, all_leaves: np.ndarray) -> float:
    # Distinct terminal leaves are disjoint; repeated class behavior is captured
    # by confidence and output_class rather than invented overlap.
    return float(np.mean(all_leaves[active] != all_leaves[active][0]))


def _node_depth(tree: object, node_id: int) -> int:
    children_left = np.asarray(getattr(tree, "children_left"), dtype=int)
    children_right = np.asarray(getattr(tree, "children_right"), dtype=int)
    stack = [(0, 0)]
    while stack:
        current, depth = stack.pop()
        if current == node_id:
            return depth
        left = int(children_left[current])
        right = int(children_right[current])
        if left >= 0:
            stack.append((left, depth + 1))
        if right >= 0:
            stack.append((right, depth + 1))
    raise ValueError(f"node not found: {node_id}")
