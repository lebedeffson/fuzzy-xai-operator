"""Preregistered confirmatory rule-ablation protocol on two real datasets."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd

from .multiclass import _stratified_cap


FROZEN_SUBGROUPS = {"uci_covertype": 3, "uci_adult": 1}


@dataclass(frozen=True)
class RuleCandidate:
    leaf_id: int
    predicted_class: int
    depth: int
    activation_rate: float
    subgroup_coverage: float
    exclusivity: float
    redundancy: float
    activation_stability: float
    score: float
    path_features: tuple[int, ...]


def run_confirmatory_rule_ablation(
    output: Path,
    cache: Path,
    *,
    seeds: Sequence[int] = (4201, 4202, 4203, 4204, 4205),
    folds: int = 10,
) -> dict[str, object]:
    from sklearn.model_selection import StratifiedKFold, train_test_split

    output.mkdir(parents=True, exist_ok=True)
    datasets = _load_datasets(cache)
    candidate_rows: list[dict[str, object]] = []
    control_rows: list[dict[str, object]] = []
    prediction_rows: list[dict[str, object]] = []
    fold_rows: list[dict[str, object]] = []
    for dataset_id, values, labels, source, license_name in datasets:
        subgroup_class = FROZEN_SUBGROUPS[dataset_id]
        raw_dataset_hash = dataset_hash(values, labels)
        for seed in seeds:
            splitter = StratifiedKFold(n_splits=folds, shuffle=True, random_state=seed)
            for fold, (train_validation, test) in enumerate(splitter.split(values, labels)):
                train, validation = train_test_split(
                    train_validation,
                    test_size=0.2,
                    random_state=seed + fold,
                    stratify=labels[train_validation],
                )
                for family in ("decision_tree", "random_forest", "gradient_boosting", "rule_ensemble_analogue"):
                    model = _fit_model(family, values[train], labels[train], seed + fold)
                    surrogate, fidelity = _fit_rule_surrogate(model, values[train], values[validation], seed + fold)
                    candidates = _describe_rules(
                        surrogate,
                        values[train],
                        labels[train] == subgroup_class,
                        values[validation],
                        labels[validation] == subgroup_class,
                    )
                    candidate = max(candidates, key=lambda item: (item.score, -item.leaf_id))
                    controls = _matched_controls(candidate, candidates, count=5)
                    base_probabilities = np.asarray(model.predict_proba(values[test]), dtype=float)
                    base_predictions = base_probabilities.argmax(axis=1)
                    test_leaves = surrogate.apply(values[test])
                    candidate_predictions = _ablate_leaf(base_probabilities, test_leaves, candidate)
                    control_effects = []
                    for control in controls:
                        control_predictions = _ablate_leaf(base_probabilities, test_leaves, control)
                        control_effects.append(
                            _subgroup_recall(labels[test], control_predictions, labels[test] == subgroup_class)
                        )
                    base_recall = _subgroup_recall(labels[test], base_predictions, labels[test] == subgroup_class)
                    candidate_recall = _subgroup_recall(
                        labels[test],
                        candidate_predictions,
                        labels[test] == subgroup_class,
                    )
                    candidate_effect = base_recall - candidate_recall
                    control_effect_values = [base_recall - value for value in control_effects]
                    specific_effect = candidate_effect - float(np.median(control_effect_values))
                    path_feature_predictions = _ablate_features(
                        model,
                        values[test],
                        candidate.path_features,
                        np.median(values[train], axis=0),
                    )
                    path_feature_effect = base_recall - _subgroup_recall(
                        labels[test],
                        path_feature_predictions,
                        labels[test] == subgroup_class,
                    )
                    related_rules = _redundancy_group(candidate, candidates)
                    group_predictions = _ablate_rule_group(base_probabilities, test_leaves, related_rules)
                    redundancy_group_effect = base_recall - _subgroup_recall(
                        labels[test],
                        group_predictions,
                        labels[test] == subgroup_class,
                    )
                    fold_id = f"{dataset_id}:{seed}:{fold}:{family}"
                    candidate_rows.append(
                        {
                            "fold_id": fold_id,
                            "dataset_id": dataset_id,
                            "seed": seed,
                            "fold": fold,
                            "family": family,
                            **candidate.__dict__,
                        }
                    )
                    control_rows.extend(
                        {
                            "fold_id": fold_id,
                            "control_index": index,
                            **control.__dict__,
                        }
                        for index, control in enumerate(controls)
                    )
                    fold_rows.append(
                        {
                            "fold_id": fold_id,
                            "dataset_id": dataset_id,
                            "source": source,
                            "license": license_name,
                            "dataset_sha256": raw_dataset_hash,
                            "seed": seed,
                            "fold": fold,
                            "family": family,
                            "subgroup_class": subgroup_class,
                            "n_train": len(train),
                            "n_validation": len(validation),
                            "n_test": len(test),
                            "surrogate_fidelity": fidelity,
                            "base_accuracy": float(np.mean(base_predictions == labels[test])),
                            "candidate_ablation_accuracy": float(np.mean(candidate_predictions == labels[test])),
                            "base_per_class_recall": json.dumps(_per_class_recall(labels[test], base_predictions)),
                            "candidate_per_class_recall": json.dumps(_per_class_recall(labels[test], candidate_predictions)),
                            "base_subgroup_macro_recall": base_recall,
                            "candidate_subgroup_macro_recall": candidate_recall,
                            "candidate_effect": candidate_effect,
                            "matched_random_effect_median": float(np.median(control_effect_values)),
                            "specific_effect": specific_effect,
                            "path_feature_effect": path_feature_effect,
                            "redundancy_group_effect": redundancy_group_effect,
                            "redundancy_group_size": len(related_rules),
                            "control_coverage_distance_mean": float(
                                np.mean([abs(control.subgroup_coverage - candidate.subgroup_coverage) for control in controls])
                            ),
                            "control_activation_distance_mean": float(
                                np.mean([abs(control.activation_rate - candidate.activation_rate) for control in controls])
                            ),
                            "test_used_for_rule_selection": False,
                        }
                    )
                    prediction_rows.extend(
                        {
                            "fold_id": fold_id,
                            "object_id": int(object_id),
                            "true_class": int(labels[object_id]),
                            "base_prediction": int(base_predictions[position]),
                            "candidate_ablation_prediction": int(candidate_predictions[position]),
                            "candidate_rule_active": bool(test_leaves[position] == candidate.leaf_id),
                            "subgroup": bool(labels[object_id] == subgroup_class),
                        }
                        for position, object_id in enumerate(test)
                    )
    fold_frame = pd.DataFrame(fold_rows)
    candidate_frame = pd.DataFrame(candidate_rows)
    control_frame = pd.DataFrame(control_rows)
    prediction_frame = pd.DataFrame(prediction_rows)
    _write_parquet(candidate_frame, output / "candidate_rules.parquet")
    _write_parquet(control_frame, output / "matched_controls.parquet")
    _write_parquet(prediction_frame, output / "object_predictions.parquet")
    fold_frame.to_csv(output / "fold_metrics.csv", index=False)
    analysis = _primary_analysis(fold_frame)
    secondary = _secondary_analysis(fold_frame, prediction_frame)
    heterogeneity = _heterogeneity(fold_frame)
    supported = bool(analysis["supported"])
    claim_status = {
        "status": "supported" if supported else "not_supported",
        "claim_removed_if_not_supported": not supported,
        "allowed_wording": (
            "The preregistered subgroup-specific rule effect exceeded matched controls on both real datasets."
            if supported
            else "Rule ablation remains a local diagnostic for a specific model and split."
        ),
        "forbidden_wording": "Rule removal has a general predictive or safety effect.",
    }
    _write_json(output / "primary_analysis.json", analysis)
    _write_json(output / "secondary_analysis.json", secondary)
    _write_json(output / "heterogeneity_report.json", heterogeneity)
    _write_json(output / "final_claim_status.json", claim_status)
    return {
        "status": claim_status["status"],
        "n_comparisons": len(fold_frame),
        "primary": analysis,
        "claim": claim_status,
    }


def _load_datasets(cache: Path) -> list[tuple[str, np.ndarray, np.ndarray, str, str]]:
    from sklearn.datasets import fetch_covtype, fetch_openml

    cache.mkdir(parents=True, exist_ok=True)
    cover = fetch_covtype(data_home=cache, download_if_missing=True)
    cover_values = np.asarray(cover.data, dtype=np.float32)
    cover_labels = np.asarray(cover.target, dtype=int) - 1
    cover_indices = _stratified_cap(np.arange(len(cover_labels)), cover_labels, 50_000, 1907)
    adult = fetch_openml("adult", version=2, data_home=cache, as_frame=True, parser="auto")
    adult_frame = adult.data.copy()
    categorical = list(adult_frame.select_dtypes(exclude="number").columns)
    adult_frame[categorical] = adult_frame[categorical].astype("string").fillna("__MISSING__")
    adult_frame = pd.get_dummies(adult_frame, columns=categorical, dummy_na=False, dtype=float)
    adult_values = adult_frame.apply(pd.to_numeric, errors="coerce").fillna(0.0).to_numpy(dtype=np.float32)
    adult_labels = np.asarray([int(str(value).strip().startswith(">50K")) for value in np.asarray(adult.target).reshape(-1)], dtype=int)
    return [
        (
            "uci_covertype",
            cover_values[cover_indices],
            cover_labels[cover_indices],
            "https://archive.ics.uci.edu/dataset/180/cover+type",
            "CC BY 4.0",
        ),
        (
            "uci_adult",
            adult_values,
            adult_labels,
            "https://archive.ics.uci.edu/dataset/2/adult",
            "CC BY 4.0",
        ),
    ]


def _fit_model(family: str, values: np.ndarray, labels: np.ndarray, seed: int) -> object:
    from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
    from sklearn.tree import DecisionTreeClassifier

    if family == "decision_tree":
        model = DecisionTreeClassifier(max_depth=12, min_samples_leaf=20, random_state=seed)
    elif family == "random_forest":
        model = RandomForestClassifier(n_estimators=60, max_depth=14, min_samples_leaf=10, n_jobs=1, random_state=seed)
    elif family == "gradient_boosting":
        model = HistGradientBoostingClassifier(max_iter=80, max_leaf_nodes=31, random_state=seed)
    else:
        model = RandomForestClassifier(n_estimators=30, max_depth=8, min_samples_leaf=20, n_jobs=1, random_state=seed)
    model.fit(values, labels)
    return model


def _fit_rule_surrogate(model: object, train: np.ndarray, validation: np.ndarray, seed: int) -> tuple[object, float]:
    from sklearn.metrics import accuracy_score
    from sklearn.tree import DecisionTreeClassifier

    surrogate = DecisionTreeClassifier(max_leaf_nodes=32, min_samples_leaf=20, random_state=seed)
    surrogate.fit(train, model.predict(train))
    fidelity = float(accuracy_score(model.predict(validation), surrogate.predict(validation)))
    return surrogate, fidelity


def _describe_rules(
    tree: object,
    train: np.ndarray,
    train_subgroup: np.ndarray,
    validation: np.ndarray,
    validation_subgroup: np.ndarray,
) -> list[RuleCandidate]:
    train_leaves = tree.apply(train)
    validation_leaves = tree.apply(validation)
    depths = _node_depths(tree.tree_)
    paths = _leaf_path_features(tree.tree_)
    result = []
    for leaf_id in np.unique(train_leaves):
        train_active = train_leaves == leaf_id
        validation_active = validation_leaves == leaf_id
        subgroup_hits = np.sum(train_active & train_subgroup)
        coverage = float(subgroup_hits / max(1, np.sum(train_subgroup)))
        exclusivity = float(subgroup_hits / max(1, np.sum(train_active)))
        train_rate = float(np.mean(train_active))
        validation_rate = float(np.mean(validation_active))
        stability = 1.0 - min(1.0, abs(train_rate - validation_rate))
        predicted_class = int(np.argmax(tree.tree_.value[int(leaf_id)][0]))
        result.append(
            RuleCandidate(
                int(leaf_id),
                predicted_class,
                int(depths[int(leaf_id)]),
                train_rate,
                coverage,
                exclusivity,
                0.0,
                stability,
                0.0,
                tuple(paths[int(leaf_id)]),
            )
        )
    if len(result) < 6:
        raise RuntimeError("rule surrogate produced fewer than six leaves")
    enriched = []
    for candidate in result:
        comparable = [
            _jaccard(candidate.path_features, other.path_features)
            for other in result
            if other.leaf_id != candidate.leaf_id and other.predicted_class == candidate.predicted_class
        ]
        redundancy = max(comparable, default=0.0)
        score = (
            0.35 * candidate.subgroup_coverage
            + 0.30 * candidate.exclusivity
            - 0.15 * redundancy
            + 0.20 * candidate.activation_stability
        )
        enriched.append(replace(candidate, redundancy=redundancy, score=score))
    return enriched


def _matched_controls(candidate: RuleCandidate, rules: Sequence[RuleCandidate], *, count: int) -> list[RuleCandidate]:
    alternatives = [rule for rule in rules if rule.leaf_id != candidate.leaf_id]
    alternatives.sort(
        key=lambda rule: (
            int(rule.predicted_class != candidate.predicted_class),
            abs(rule.subgroup_coverage - candidate.subgroup_coverage),
            abs(rule.activation_rate - candidate.activation_rate),
            abs(rule.depth - candidate.depth),
            abs(rule.redundancy - candidate.redundancy),
            abs(rule.score - candidate.score),
            rule.leaf_id,
        )
    )
    if len(alternatives) < count:
        raise RuntimeError("not enough matched control rules")
    return alternatives[:count]


def _ablate_leaf(probabilities: np.ndarray, leaves: np.ndarray, rule: RuleCandidate) -> np.ndarray:
    adjusted = probabilities.copy()
    active = leaves == rule.leaf_id
    adjusted[active, rule.predicted_class] = 0.0
    totals = adjusted[active].sum(axis=1, keepdims=True)
    adjusted[active] = np.divide(adjusted[active], totals, out=np.zeros_like(adjusted[active]), where=totals > 0)
    return adjusted.argmax(axis=1)


def _ablate_rule_group(
    probabilities: np.ndarray,
    leaves: np.ndarray,
    rules: Sequence[RuleCandidate],
) -> np.ndarray:
    adjusted = probabilities.copy()
    changed = np.zeros(len(leaves), dtype=bool)
    for rule in rules:
        active = leaves == rule.leaf_id
        adjusted[active, rule.predicted_class] = 0.0
        changed |= active
    totals = adjusted[changed].sum(axis=1, keepdims=True)
    adjusted[changed] = np.divide(
        adjusted[changed],
        totals,
        out=np.zeros_like(adjusted[changed]),
        where=totals > 0,
    )
    return adjusted.argmax(axis=1)


def _ablate_features(model: object, values: np.ndarray, features: Sequence[int], reference: np.ndarray) -> np.ndarray:
    modified = values.copy()
    selected = np.asarray(sorted(set(features)), dtype=int)
    if len(selected):
        modified[:, selected] = reference[selected]
    return np.asarray(model.predict(modified), dtype=int)


def _redundancy_group(candidate: RuleCandidate, rules: Sequence[RuleCandidate]) -> list[RuleCandidate]:
    related = [
        rule
        for rule in rules
        if rule.predicted_class == candidate.predicted_class
        and _jaccard(candidate.path_features, rule.path_features) >= 0.5
    ]
    return related or [candidate]


def _jaccard(left: Sequence[int], right: Sequence[int]) -> float:
    left_set = set(left)
    right_set = set(right)
    union = left_set | right_set
    return len(left_set & right_set) / len(union) if union else 0.0


def _subgroup_recall(labels: np.ndarray, predictions: np.ndarray, subgroup: np.ndarray) -> float:
    from sklearn.metrics import recall_score

    if not subgroup.any():
        return 0.0
    return float(recall_score(labels[subgroup], predictions[subgroup], average="macro", zero_division=0))


def _per_class_recall(labels: np.ndarray, predictions: np.ndarray) -> list[float]:
    from sklearn.metrics import recall_score

    classes = np.unique(labels)
    return [float(value) for value in recall_score(labels, predictions, labels=classes, average=None, zero_division=0)]


def _primary_analysis(frame: pd.DataFrame) -> dict[str, object]:
    from scipy.stats import ttest_1samp

    values = frame["specific_effect"].to_numpy(dtype=float)
    interval = _cluster_bootstrap(frame, repetitions=5000, seed=4201)
    dataset_effects = frame.groupby("dataset_id")["specific_effect"].mean().to_dict()
    dataset_results = []
    raw_p_values = []
    for dataset_id, group in frame.groupby("dataset_id"):
        seed_means = group.groupby("seed")["specific_effect"].mean().to_numpy(dtype=float)
        statistic = ttest_1samp(seed_means, 0.0, alternative="greater")
        raw_p = float(statistic.pvalue) if np.isfinite(statistic.pvalue) else 1.0
        raw_p_values.append(raw_p)
        dataset_results.append(
            {
                "dataset_id": dataset_id,
                "mean_specific_effect": float(seed_means.mean()),
                "standardized_effect": float(seed_means.mean() / seed_means.std(ddof=1)) if seed_means.std(ddof=1) > 0 else 0.0,
                "one_sided_cluster_p": raw_p,
            }
        )
    corrected = _holm(raw_p_values)
    for row, corrected_p in zip(dataset_results, corrected):
        row["holm_corrected_p"] = corrected_p
    supported = (
        interval[0] > 0.01
        and all(value > 0.0 for value in dataset_effects.values())
        and all(float(row["holm_corrected_p"]) <= 0.05 for row in dataset_results)
    )
    return {
        "endpoint": "subgroup_macro_recall_specific_effect",
        "n_paired_comparisons": len(frame),
        "mean_specific_effect": float(values.mean()),
        "confidence_interval_95": interval,
        "practically_null_interval": [-0.01, 0.01],
        "dataset_effects": dataset_effects,
        "dataset_replications": dataset_results,
        "corrected_p_method": "Holm correction over frozen dataset replications; seed-cluster means",
        "supported": supported,
        "selection_uses_test": False,
    }


def _secondary_analysis(frame: pd.DataFrame, predictions: pd.DataFrame) -> dict[str, object]:
    changed = predictions["base_prediction"] != predictions["candidate_ablation_prediction"]
    return {
        "candidate_effect_mean": float(frame["candidate_effect"].mean()),
        "matched_random_effect_mean": float(frame["matched_random_effect_median"].mean()),
        "prediction_change_rate": float(changed.mean()),
        "path_feature_effect_mean": float(frame["path_feature_effect"].mean()),
        "redundancy_group_effect_mean": float(frame["redundancy_group_effect"].mean()),
        "control_coverage_distance_mean": float(frame["control_coverage_distance_mean"].mean()),
        "control_activation_distance_mean": float(frame["control_activation_distance_mean"].mean()),
        "holm_correction": "reserved for multiple secondary endpoints; no secondary significance claim",
        "variants": {
            "candidate_rule": "measured",
            "matched_random_rules": "measured_five_per_candidate",
            "decision_path": "represented by the selected surrogate leaf path",
            "candidate_feature": "measured by replacing selected path features with train-reference medians",
            "redundancy_groups": "measured for same-class rules sharing at least half of path features",
        },
    }


def _heterogeneity(frame: pd.DataFrame) -> dict[str, object]:
    grouped = frame.groupby(["dataset_id", "family"])["specific_effect"].agg(["mean", "std", "count"])
    return {
        "by_dataset_and_family": [
            {
                "dataset_id": dataset,
                "family": family,
                "mean": float(row["mean"]),
                "std": None if np.isnan(row["std"]) else float(row["std"]),
                "count": int(row["count"]),
            }
            for (dataset, family), row in grouped.iterrows()
        ],
        "mixed_effects_status": "descriptive cluster effects plus hierarchical bootstrap; no unverified p-value claim",
    }


def _cluster_bootstrap(frame: pd.DataFrame, *, repetitions: int, seed: int) -> list[float]:
    rng = np.random.default_rng(seed)
    clusters = [group["specific_effect"].to_numpy(dtype=float) for _, group in frame.groupby(["dataset_id", "seed"])]
    means = []
    for _ in range(repetitions):
        selected = rng.integers(0, len(clusters), size=len(clusters))
        values = np.concatenate([clusters[index] for index in selected])
        means.append(float(values.mean()))
    return [float(np.quantile(means, 0.025)), float(np.quantile(means, 0.975))]


def _holm(values: Sequence[float]) -> list[float]:
    order = np.argsort(values)
    adjusted = np.zeros(len(values), dtype=float)
    running = 0.0
    for rank, index in enumerate(order):
        current = min(1.0, (len(values) - rank) * float(values[int(index)]))
        running = max(running, current)
        adjusted[int(index)] = running
    return adjusted.tolist()


def _node_depths(tree: object) -> np.ndarray:
    depths = np.zeros(tree.node_count, dtype=int)
    stack = [(0, 0)]
    while stack:
        node, depth = stack.pop()
        depths[node] = depth
        left = int(tree.children_left[node])
        right = int(tree.children_right[node])
        if left != right:
            stack.extend(((left, depth + 1), (right, depth + 1)))
    return depths


def _leaf_path_features(tree: object) -> dict[int, list[int]]:
    result: dict[int, list[int]] = {}
    stack = [(0, [])]
    while stack:
        node, path = stack.pop()
        left = int(tree.children_left[node])
        right = int(tree.children_right[node])
        if left == right:
            result[node] = path
            continue
        feature = int(tree.feature[node])
        stack.extend(((left, [*path, feature]), (right, [*path, feature])))
    return result


def _write_parquet(frame: pd.DataFrame, path: Path) -> None:
    frame.to_parquet(path, index=False)


def _write_json(path: Path, payload: object) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def dataset_hash(values: np.ndarray, labels: np.ndarray) -> str:
    digest = hashlib.sha256()
    digest.update(np.ascontiguousarray(values).tobytes())
    digest.update(np.ascontiguousarray(labels).tobytes())
    return digest.hexdigest()
