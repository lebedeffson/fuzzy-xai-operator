from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import platform
import subprocess
from dataclasses import replace
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
from sklearn.base import clone
from sklearn.cluster import KMeans
from sklearn.datasets import load_breast_cancer
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier

from fuzzyxai import ExplainPlan, FuzzyXAI
from fuzzyxai.adapters import SklearnAdapter
from fuzzyxai.evidence import (
    ExplanationEvidence,
    LearnedRule,
    RuleAblationEvidence,
    TrainingCheckpointEvidence,
    build_object_trace,
    evaluate_rule_ablation,
    find_similar_tabular_cases,
    find_tabular_counterfactuals,
    select_explanatory_cases,
    validate_domain_language,
)


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUT = ROOT / "release_evidence/empirical_experiments/breast_cancer_checkpoint"
SEED = 42
EPOCHS = 30
RUN_ID = "bcwd_sgd_checkpoint_seed42_v1"
DATASET_URL = "https://archive.ics.uci.edu/dataset/17/breast+cancer+wisconsin+diagnostic"
DATASET_DOI = "10.24432/C5DW2B"
PROTOCOL_TIME = "2026-07-19T00:00:00+00:00"


def _json_default(value: Any) -> Any:
    if hasattr(value, "tolist"):
        return value.tolist()
    if isinstance(value, (np.integer, np.floating)):
        return value.item()
    raise TypeError(f"cannot serialize {type(value).__name__}")


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True, default=_json_default) + "\n",
        encoding="utf-8",
    )


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def sha256_file(path: Path) -> str:
    return sha256_bytes(path.read_bytes())


def model_fingerprint(model: Any) -> str:
    digest = hashlib.sha256(f"{type(model).__module__}.{type(model).__qualname__}".encode())
    for name in ("coef_", "intercept_", "classes_", "cluster_centers_", "consequents_"):
        value = getattr(model, name, None)
        if value is not None:
            digest.update(name.encode())
            digest.update(np.asarray(value).tobytes())
    getter = getattr(model, "get_params", None)
    if callable(getter):
        digest.update(repr(sorted(getter(deep=False).items())).encode())
    return digest.hexdigest()


def expected_calibration_error(labels: np.ndarray, probabilities: np.ndarray, bins: int = 10) -> float:
    positive = np.asarray(probabilities, dtype=float)[:, 1]
    edges = np.linspace(0.0, 1.0, bins + 1)
    result = 0.0
    for lower, upper in zip(edges[:-1], edges[1:]):
        mask = (positive >= lower) & (positive < upper if upper < 1.0 else positive <= upper)
        if not mask.any():
            continue
        result += float(mask.mean()) * abs(float(positive[mask].mean()) - float(labels[mask].mean()))
    return result


def metrics(labels: np.ndarray, probabilities: np.ndarray, subgroup_mask: np.ndarray) -> dict[str, float]:
    predictions = np.argmax(probabilities, axis=1)
    subgroup_labels = labels[subgroup_mask]
    subgroup_predictions = predictions[subgroup_mask]
    return {
        "accuracy": float(accuracy_score(labels, predictions)),
        "recall": float(recall_score(labels, predictions, zero_division=0)),
        "precision": float(precision_score(labels, predictions, zero_division=0)),
        "calibration_error": expected_calibration_error(labels, probabilities),
        "subgroup_accuracy": float(accuracy_score(subgroup_labels, subgroup_predictions)) if subgroup_mask.any() else math.nan,
        "subgroup_recall": float(recall_score(subgroup_labels, subgroup_predictions, zero_division=0)) if subgroup_mask.any() else math.nan,
        "subgroup_precision": float(precision_score(subgroup_labels, subgroup_predictions, zero_division=0)) if subgroup_mask.any() else math.nan,
        "critical_errors": float(np.sum((labels == 1) & (predictions == 0))),
    }


class MeasuredSugenoRuleClassifier:
    """Data-fitted fixed-premise Sugeno rule baseline with native rule disclosure."""

    def __init__(self, n_rules: int = 5, random_state: int = SEED):
        self.n_rules = n_rules
        self.random_state = random_state
        self.suppressed_rules: set[int] = set()

    def get_params(self, deep: bool = True) -> dict[str, Any]:
        return {"n_rules": self.n_rules, "random_state": self.random_state}

    def fit(self, values: Sequence[Sequence[float]], labels: Sequence[int]) -> "MeasuredSugenoRuleClassifier":
        matrix = np.asarray(values, dtype=float)
        target = np.asarray(labels, dtype=int)
        self.classes_ = np.asarray([0, 1])
        clustering = KMeans(self.n_rules, random_state=self.random_state, n_init=20).fit(matrix)
        self.cluster_centers_ = clustering.cluster_centers_
        global_scale = np.std(matrix, axis=0)
        scales: list[np.ndarray] = []
        consequents: list[float] = []
        rules: list[dict[str, Any]] = []
        for rule_index in range(self.n_rules):
            members = clustering.labels_ == rule_index
            local = matrix[members]
            scale = np.std(local, axis=0) if len(local) > 1 else global_scale
            scale = np.where(scale > 0.15, scale, np.where(global_scale > 0.15, global_scale, 1.0))
            consequent = float(target[members].mean()) if members.any() else float(target.mean())
            scales.append(scale)
            consequents.append(consequent)
            rules.append(
                {
                    "rule_id": f"sugeno_{rule_index:02d}",
                    "antecedents": [f"cluster_distance_to_center_{rule_index} is low"],
                    "consequent": str(int(consequent >= 0.5)),
                    "coverage": float(members.mean()),
                    "precision": float(max(consequent, 1.0 - consequent)),
                    "support": int(members.sum()),
                    "stability": None,
                    "importance": None,
                    "human_text": "Gaussian premise learned from the training partition",
                    "complexity": float(matrix.shape[1]),
                    "evidence_refs": [f"measured_training_cluster:{rule_index}"],
                }
            )
        self.scales_ = np.asarray(scales)
        self.consequents_ = np.asarray(consequents)
        self.rules_ = rules
        return self

    def _activations(self, values: Sequence[Sequence[float]]) -> np.ndarray:
        matrix = np.asarray(values, dtype=float)
        delta = (matrix[:, None, :] - self.cluster_centers_[None, :, :]) / self.scales_[None, :, :]
        activation = np.exp(-0.5 * np.mean(delta**2, axis=2))
        for index in self.suppressed_rules:
            activation[:, index] = 0.0
        denominator = activation.sum(axis=1, keepdims=True)
        return np.divide(activation, denominator, out=np.full_like(activation, 1.0 / self.n_rules), where=denominator > 1e-12)

    def predict_proba(self, values: Sequence[Sequence[float]]) -> np.ndarray:
        positive = np.clip(self._activations(values) @ self.consequents_, 1e-6, 1.0 - 1e-6)
        return np.column_stack((1.0 - positive, positive))

    def predict(self, values: Sequence[Sequence[float]]) -> np.ndarray:
        return (self.predict_proba(values)[:, 1] >= 0.5).astype(int)


def load_dataset_snapshot(output: Path) -> tuple[np.ndarray, np.ndarray, list[str], np.ndarray, Path]:
    dataset = load_breast_cancer()
    values = np.asarray(dataset.data, dtype=float)
    labels = 1 - np.asarray(dataset.target, dtype=int)  # 1 is the malignant dataset label.
    names = [str(item) for item in dataset.feature_names]
    object_ids = np.asarray([f"bcwd_{index:04d}" for index in range(len(labels))])
    snapshot = output / "data/dataset_snapshot.csv"
    snapshot.parent.mkdir(parents=True, exist_ok=True)
    with snapshot.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle, lineterminator="\n")
        writer.writerow(["object_id", *names, "target_malignant_dataset_label"])
        for object_id, row, label in zip(object_ids, values, labels):
            writer.writerow([object_id, *[format(float(item), ".12g") for item in row], int(label)])
    return values, labels, names, object_ids, snapshot


def split_indices(labels: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    indices = np.arange(len(labels))
    train, remainder = train_test_split(indices, test_size=0.4, random_state=SEED, stratify=labels)
    validation, test = train_test_split(remainder, test_size=0.5, random_state=SEED, stratify=labels[remainder])
    return np.sort(train), np.sort(validation), np.sort(test)


def subgroup_definition(
    standardized: np.ndarray,
    train: np.ndarray,
) -> tuple[np.ndarray, dict[str, Any]]:
    clustering = KMeans(n_clusters=3, random_state=SEED, n_init=20).fit(standardized[train])
    train_counts = np.bincount(clustering.labels_, minlength=3)
    selected_cluster = int(np.argmin(train_counts))
    assignments = clustering.predict(standardized)
    mask = assignments == selected_cluster
    definition = {
        "subgroup_id": "rare_unsupervised_cluster_001",
        "definition_time": "before checkpoint-model training and before forgetting-case selection",
        "method": "smallest KMeans cluster on standardized train features",
        "uses_test_labels": False,
        "n_clusters": 3,
        "selected_cluster": selected_cluster,
        "train_cluster_sizes": train_counts.tolist(),
        "scaler_fit_partition": "train",
        "random_seed": SEED,
        "centroids_sha256": sha256_bytes(np.asarray(clustering.cluster_centers_).tobytes()),
    }
    definition["subgroup_definition_hash"] = sha256_bytes(
        json.dumps(definition, sort_keys=True, separators=(",", ":")).encode()
    )
    return mask, definition


def train_checkpoint_model(
    values: np.ndarray,
    labels: np.ndarray,
    object_ids: np.ndarray,
    train: np.ndarray,
    validation: np.ndarray,
    test: np.ndarray,
    subgroup: np.ndarray,
    scaler: StandardScaler,
) -> tuple[SGDClassifier, list[TrainingCheckpointEvidence], dict[str, list[dict[str, Any]]]]:
    standardized = scaler.transform(values)
    model = SGDClassifier(
        loss="log_loss",
        alpha=0.0005,
        learning_rate="constant",
        eta0=0.01,
        random_state=SEED,
    )
    rng = np.random.default_rng(SEED)
    checkpoints: list[TrainingCheckpointEvidence] = []
    histories = {str(object_ids[index]): [] for index in validation}
    nearest = {}
    train_matrix = standardized[train]
    for index in validation:
        distances = np.linalg.norm(train_matrix - standardized[index], axis=1)
        nearest[str(object_ids[index])] = [str(object_ids[train[position]]) for position in np.argsort(distances)[:3]]

    for epoch in range(1, EPOCHS + 1):
        order = train.copy()
        rng.shuffle(order)
        model.partial_fit(standardized[order], labels[order], classes=np.asarray([0, 1]) if epoch == 1 else None)
        partition_metrics = {
            "train": metrics(labels[train], model.predict_proba(standardized[train]), subgroup[train]),
            "validation": metrics(labels[validation], model.predict_proba(standardized[validation]), subgroup[validation]),
            "test": metrics(labels[test], model.predict_proba(standardized[test]), subgroup[test]),
        }
        validation_probabilities = model.predict_proba(standardized[validation])
        validation_predictions = np.argmax(validation_probabilities, axis=1)
        prediction_map: dict[str, int] = {}
        confidence_map: dict[str, float] = {}
        loss_map: dict[str, float] = {}
        margin_map: dict[str, float] = {}
        active_map: dict[str, Sequence[str]] = {}
        for local_index, source_index in enumerate(validation):
            object_id = str(object_ids[source_index])
            truth = int(labels[source_index])
            probability = validation_probabilities[local_index]
            confidence = float(probability[truth])
            prediction = int(validation_predictions[local_index])
            local_terms = np.asarray(model.coef_[0]) * standardized[source_index]
            top_features = np.argsort(np.abs(local_terms))[::-1][:3]
            active = [f"linear_term:{int(item)}" for item in top_features]
            item = {
                "epoch": epoch,
                "predicted_class": prediction,
                "confidence": confidence,
                "loss": float(-math.log(max(confidence, 1e-12))),
                "margin": float(confidence - float(np.max(np.delete(probability, truth)))),
                "correct": prediction == truth,
                "rule_activations": {name: float(abs(local_terms[int(name.rsplit(':', 1)[1])])) for name in active},
                "global_metric": partition_metrics["validation"]["accuracy"],
                "subgroup_metric": partition_metrics["validation"]["subgroup_accuracy"],
            }
            histories[object_id].append(item)
            prediction_map[object_id] = prediction
            confidence_map[object_id] = confidence
            loss_map[object_id] = float(item["loss"])
            margin_map[object_id] = float(item["margin"])
            active_map[object_id] = active
        checkpoints.append(
            TrainingCheckpointEvidence(
                run_id=RUN_ID,
                checkpoint_id=f"checkpoint_{epoch:03d}",
                epoch=epoch,
                model_fingerprint=model_fingerprint(model),
                train_metric=partition_metrics["train"]["accuracy"],
                validation_metric=partition_metrics["validation"]["accuracy"],
                test_metric=partition_metrics["test"]["accuracy"],
                subgroup_metric=partition_metrics["validation"]["subgroup_accuracy"],
                object_predictions=prediction_map,
                object_confidences=confidence_map,
                object_losses=loss_map,
                object_margins=margin_map,
                active_rules=active_map,
                nearest_neighbors=nearest,
                random_seed=SEED,
                captured_at=f"{PROTOCOL_TIME}#epoch-{epoch:03d}",
            )
        )
    return model, checkpoints, histories


def select_forgetting_case(
    histories: Mapping[str, Sequence[Mapping[str, Any]]],
    object_ids: np.ndarray,
    validation: np.ndarray,
    subgroup: np.ndarray,
) -> tuple[str, str, list[dict[str, Any]]]:
    source_index_by_id = {str(object_ids[index]): int(index) for index in validation}
    candidates: list[tuple[int, float, int, str, list[dict[str, Any]]]] = []
    for object_id, raw_history in histories.items():
        history = [dict(item) for item in raw_history]
        events = [
            int(current["epoch"])
            for previous, current in zip(history, history[1:])
            if bool(previous["correct"]) and not bool(current["correct"])
        ]
        if not events:
            continue
        max_drop = max(float(left["confidence"]) - float(right["confidence"]) for left, right in zip(history, history[1:]))
        source_index = source_index_by_id[object_id]
        candidates.append((len(events), max_drop, int(subgroup[source_index]), object_id, history))
    if not candidates:
        raise RuntimeError("no validation forgetting event was found; protocol seed/model must not silently fabricate one")
    candidates.sort(key=lambda item: (-item[0], -item[1], -item[2], item[3]))
    _, _, _, source_id, selected_history = candidates[0]
    return "case_real_001", source_id, selected_history


def tree_rule_conditions(model: DecisionTreeClassifier, leaf: int, feature_names: Sequence[str]) -> tuple[list[str], int]:
    tree = model.tree_
    parent: dict[int, tuple[int, str]] = {}
    for node, (left, right) in enumerate(zip(tree.children_left, tree.children_right)):
        if left == right:
            continue
        feature = feature_names[int(tree.feature[node])]
        threshold = float(tree.threshold[node])
        parent[int(left)] = (node, f"{feature} <= {threshold:.6g}")
        parent[int(right)] = (node, f"{feature} > {threshold:.6g}")
    conditions: list[str] = []
    current = leaf
    while current in parent:
        current, condition = parent[current]
        conditions.append(condition)
    return list(reversed(conditions)), current


def suppress_tree_leaf_probabilities(model: DecisionTreeClassifier, values: np.ndarray, target_leaf: int) -> np.ndarray:
    tree = model.tree_
    parent: dict[int, int] = {}
    sibling: dict[int, int] = {}
    for node, (left, right) in enumerate(zip(tree.children_left, tree.children_right)):
        if left == right:
            continue
        parent[int(left)] = node
        parent[int(right)] = node
        sibling[int(left)] = int(right)
        sibling[int(right)] = int(left)
    if target_leaf not in parent:
        raise ValueError("cannot suppress the root leaf")
    fallback_node = sibling[target_leaf]
    fallback = np.asarray(tree.value[fallback_node], dtype=float).reshape(-1)
    fallback /= fallback.sum()
    probabilities = model.predict_proba(values)
    mask = model.apply(values) == target_leaf
    probabilities[mask] = fallback
    return probabilities


def measured_tree_ablation(
    standardized: np.ndarray,
    labels: np.ndarray,
    names: Sequence[str],
    train: np.ndarray,
    validation: np.ndarray,
    test: np.ndarray,
    subgroup: np.ndarray,
    target_index: int,
) -> tuple[RuleAblationEvidence, LearnedRule, DecisionTreeClassifier]:
    model = DecisionTreeClassifier(max_depth=4, min_samples_leaf=3, random_state=SEED).fit(standardized[train], labels[train])
    target_leaf = int(model.apply(standardized[[target_index]])[0])
    conditions, _ = tree_rule_conditions(model, target_leaf, names)
    with_metrics = {
        split: metrics(labels[indices], model.predict_proba(standardized[indices]), subgroup[indices])
        for split, indices in (("train", train), ("validation", validation), ("test", test))
    }
    without_metrics = {
        split: metrics(
            labels[indices],
            suppress_tree_leaf_probabilities(model, standardized[indices], target_leaf),
            subgroup[indices],
        )
        for split, indices in (("train", train), ("validation", validation), ("test", test))
    }
    before_target = model.predict_proba(standardized[[target_index]])
    after_target = suppress_tree_leaf_probabilities(model, standardized[[target_index]], target_leaf)
    evidence = RuleAblationEvidence(
        run_id=RUN_ID,
        rule_id=f"tree_leaf_{target_leaf}",
        model_fingerprint=model_fingerprint(model),
        native=True,
        surrogate=False,
        fidelity=None,
        train_metrics_with_rule=with_metrics["train"],
        validation_metrics_with_rule=with_metrics["validation"],
        test_metrics_with_rule=with_metrics["test"],
        train_metrics_without_rule=without_metrics["train"],
        validation_metrics_without_rule=without_metrics["validation"],
        test_metrics_without_rule=without_metrics["test"],
        subgroup_metrics_with_rule={key: value for key, value in with_metrics["validation"].items() if key.startswith("subgroup_")},
        subgroup_metrics_without_rule={key: value for key, value in without_metrics["validation"].items() if key.startswith("subgroup_")},
        critical_errors_with_rule=int(with_metrics["test"]["critical_errors"]),
        critical_errors_without_rule=int(without_metrics["test"]["critical_errors"]),
        target_prediction_with_rule=int(np.argmax(before_target[0])),
        target_prediction_without_rule=int(np.argmax(after_target[0])),
        limitations=(
            "tree leaf suppression uses the sibling branch distribution as the explicit fallback",
            "the measured effect applies to this split, seed, and tree configuration",
        ),
    )
    support = int(np.sum(model.apply(standardized[train]) == target_leaf))
    rule = LearnedRule(
        rule_id=evidence.rule_id,
        model_version=evidence.model_fingerprint[:12],
        antecedents=conditions,
        consequent=str(evidence.target_prediction_with_rule),
        activation=1.0,
        coverage=support / len(train),
        precision=None,
        support=support,
        stability=None,
        importance=None,
        counterfactual_effect={},
        source_objects=(),
        class_distribution={},
        human_text=" and ".join(conditions) + f" -> class {evidence.target_prediction_with_rule}",
        complexity=float(len(conditions)),
        is_primary=True,
        is_redundant=False,
        is_conflicting=False,
        native=True,
        surrogate=False,
        evidence_refs=(f"model.tree_.leaf:{target_leaf}", "measured_sibling_fallback_ablation"),
    )
    baseline = {"train": with_metrics["train"]["accuracy"], "validation": with_metrics["validation"]["accuracy"], "test": with_metrics["test"]["accuracy"], "subgroup_recall": with_metrics["validation"]["subgroup_recall"], "critical_errors": with_metrics["test"]["critical_errors"], "calibration": with_metrics["test"]["calibration_error"]}
    ablated = {"train": without_metrics["train"]["accuracy"], "validation": without_metrics["validation"]["accuracy"], "test": without_metrics["test"]["accuracy"], "subgroup_recall": without_metrics["validation"]["subgroup_recall"], "critical_errors": without_metrics["test"]["critical_errors"], "calibration": without_metrics["test"]["calibration_error"]}
    return evidence, evaluate_rule_ablation(rule, baseline_metrics=baseline, ablated_metrics=ablated), model


def cross_model_matrix(
    standardized: np.ndarray,
    labels: np.ndarray,
    names: Sequence[str],
    train: np.ndarray,
    target_index: int,
) -> list[dict[str, Any]]:
    models: list[tuple[str, Any, str]] = [
        ("logistic_regression", LogisticRegression(max_iter=2000, random_state=SEED), "sklearn"),
        ("decision_tree", DecisionTreeClassifier(max_depth=4, min_samples_leaf=3, random_state=SEED), "sklearn"),
        ("random_forest", RandomForestClassifier(n_estimators=80, max_depth=5, random_state=SEED), "sklearn"),
        ("sugeno_native_rules", MeasuredSugenoRuleClassifier(n_rules=5, random_state=SEED), "native_rules"),
    ]
    rows: list[dict[str, Any]] = []
    for name, model, adapter in models:
        fitted = model.fit(standardized[train], labels[train])
        result = FuzzyXAI.wrap(fitted, adapter=adapter).explain_one(
            standardized[target_index],
            object_id="case_real_001",
            feature_names=names,
            reference_data=standardized[train],
            reference_labels=labels[train].tolist(),
            include_model_knowledge=True,
        )
        capability = result.view_model.trace["adapter_capabilities"]
        rules = result.view_model.layers.get("rules", [])
        rows.append(
            {
                "model": name,
                "model_fingerprint": model_fingerprint(fitted),
                "prediction": int(fitted.predict(standardized[[target_index]])[0]),
                "explanation_level": result.explanation_level,
                "native_channels": list(result.native_channels),
                "surrogate_channels": list(result.surrogate_channels),
                "missing_channels": list(result.missing_channels),
                "adapter_capabilities": capability,
                "native_rule_count": len([rule for rule in rules if rule.get("native")]),
                "graph_errors": list(result.explanation_graph.validate_reachability()),
                "action": result.action,
            }
        )

    fitted_logistic = clone(models[0][1]).fit(standardized[train], labels[train])

    def black_box(rows: Sequence[Sequence[float]]) -> list[int]:
        return fitted_logistic.predict(np.asarray(rows, dtype=float)).tolist()

    black_result = FuzzyXAI.wrap(black_box, adapter="callable").explain_one(
        standardized[target_index],
        object_id="case_real_001",
        feature_names=names,
        reference_data=standardized[train],
        reference_labels=labels[train].tolist(),
    )
    rows.append(
        {
            "model": "black_box_callable",
            "model_fingerprint": sha256_bytes(repr(fitted_logistic.get_params()).encode()),
            "prediction": int(black_box([standardized[target_index]])[0]),
            "explanation_level": black_result.explanation_level,
            "native_channels": list(black_result.native_channels),
            "surrogate_channels": list(black_result.surrogate_channels),
            "missing_channels": list(black_result.missing_channels),
            "adapter_capabilities": black_result.view_model.trace["adapter_capabilities"],
            "native_rule_count": len(
                [rule for rule in black_result.view_model.layers.get("rules", []) if rule.get("native")]
            ),
            "graph_errors": list(black_result.explanation_graph.validate_reachability()),
            "action": black_result.action,
        }
    )
    return rows


def domain_plan() -> ExplainPlan:
    plan = ExplainPlan.default()
    plan.domain_language = {
        "version": "bcwd-neutral-methodological-v1",
        "scope": "medical_research",
        "features": {
            "worst concave points": {
                "label": "максимальная выраженность вогнутых участков контура",
                "meaning": "измеренный геометрический признак ядер клеток в наборе BCDW",
                "expected_direction": "unknown",
                "expert_review_status": "not_reviewed",
            },
            "worst perimeter": {
                "label": "максимальный периметр",
                "meaning": "измеренный размер контура ядер клеток в наборе BCDW",
                "expected_direction": "unknown",
                "expert_review_status": "not_reviewed",
            },
        },
        "classes": {
            0: {"label": "метка benign в исследовательском наборе", "domain_defined": True},
            1: {"label": "метка malignant в исследовательском наборе", "domain_defined": True},
        },
        "actions": {
            "review": {
                "label": "Проверить исследовательский результат",
                "explanation": "Не использовать классификацию как медицинское заключение; проверить evidence и ограничения протокола.",
            }
        },
    }
    return plan


def build_dataset_card(
    output: Path,
    snapshot: Path,
    values: np.ndarray,
    names: Sequence[str],
    train: np.ndarray,
    validation: np.ndarray,
    test: np.ndarray,
    subgroup_definition_payload: Mapping[str, Any],
) -> None:
    card = f"""# Dataset card: Breast Cancer Wisconsin (Diagnostic)

- Source: {DATASET_URL}
- DOI: {DATASET_DOI}
- License: Creative Commons Attribution 4.0 International (CC BY 4.0)
- Bundled loader: `sklearn.datasets.load_breast_cancer`
- Version anchor: scikit-learn bundled copy used by this environment
- Objects: {len(values)}
- Features: {len(names)}
- Target: UCI diagnostic dataset label, recoded as `1=malignant`, `0=benign`
- Missing values in bundled matrix: {int(np.isnan(values).sum())}
- Split: train={len(train)}, validation={len(validation)}, test={len(test)}
- Split seed: {SEED}
- Preprocessing: `StandardScaler` fitted on train only
- Rare subgroup: smallest of three KMeans clusters fitted on standardized train features before model training
- Subgroup definition hash: `{subgroup_definition_payload['subgroup_definition_hash']}`
- Snapshot SHA256: `{sha256_file(snapshot)}`
- Download/access date recorded by protocol: 2026-07-19

## Limitations

This is a methodological classification benchmark. The experiment does not establish clinical validity,
diagnostic utility, fairness, or deployment readiness. Human-readable domain wording remains unavailable
until an independent subject-matter reviewer signs the versioned dictionary.
"""
    (output / "dataset_card.md").write_text(card, encoding="utf-8")


def run(output_dir: str | Path = DEFAULT_OUTPUT) -> dict[str, Any]:
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    values, labels, names, object_ids, snapshot = load_dataset_snapshot(output)
    train, validation, test = split_indices(labels)
    scaler = StandardScaler().fit(values[train])
    standardized = scaler.transform(values)
    subgroup, subgroup_payload = subgroup_definition(standardized, train)
    write_json(output / "subgroup_definition.json", subgroup_payload)
    build_dataset_card(output, snapshot, values, names, train, validation, test, subgroup_payload)

    checkpoint_model, checkpoints, histories = train_checkpoint_model(
        values, labels, object_ids, train, validation, test, subgroup, scaler
    )
    public_id, source_id, selected_history = select_forgetting_case(histories, object_ids, validation, subgroup)
    source_index = int(np.where(object_ids == source_id)[0][0])
    trace = build_object_trace(public_id, selected_history, confidence_drop=1.0)
    checkpoint_payload = [checkpoint.to_dict() for checkpoint in checkpoints]
    write_json(output / "checkpoints.json", checkpoint_payload)
    write_json(
        output / "selected_forgetting_case.json",
        {
            "public_id": public_id,
            "source_dataset_object_id": source_id,
            "source_index": source_index,
            "selection_partition": "validation",
            "selection_algorithm": "descending label-loss event count, confidence drop, pre-defined subgroup membership, stable object id",
            "selected_after_training": True,
            "forgetting_events": list(trace.forgetting_events),
            "first_learned_epoch": trace.first_learned_epoch,
            "last_correct_epoch": trace.last_correct_epoch,
            "rare_subgroup_member": bool(subgroup[source_index]),
            "result_origin": "measured",
            "history": selected_history,
        },
    )

    ablation, learned_rule, tree_model = measured_tree_ablation(
        standardized, labels, names, train, validation, test, subgroup, source_index
    )
    write_json(output / "rule_ablation.json", ablation.to_dict())

    nearest = find_similar_tabular_cases(
        standardized[source_index],
        standardized[train],
        query_object_id=public_id,
        reference_ids=[str(item) for item in object_ids[train]],
        feature_names=names,
        reference_labels=labels[train].tolist(),
        reference_predictions=checkpoint_model.predict(standardized[train]).tolist(),
        limit=len(train),
    )
    selected_cases = select_explanatory_cases(
        nearest,
        predicted_label=int(checkpoint_model.predict(standardized[[source_index]])[0]),
    )
    write_json(output / "similar_cases.json", [item.to_dict() for item in selected_cases])

    prediction_pipeline = Pipeline([("scaler", scaler), ("model", checkpoint_model)])
    sensitivity = find_tabular_counterfactuals(
        SklearnAdapter(prediction_pipeline),
        values[source_index],
        values[train],
        feature_names=names,
        limit=1,
    )
    write_json(output / "sensitivity_analysis.json", [item.to_dict() for item in sensitivity])

    cross_model = cross_model_matrix(standardized, labels, names, train, source_index)
    write_json(output / "cross_model_matrix.json", cross_model)

    plan = domain_plan()
    semantic = validate_domain_language(plan.domain_language, regulated_domain=True)
    write_json(output / "domain_language_validation.json", semantic.to_dict())
    plan_payload = plan.to_dict()
    write_json(output / "domain_language.json", plan_payload["domain_language"])

    training = FuzzyXAI.wrap(checkpoint_model, adapter="sklearn", explain_plan=plan).observe_training(
        history={
            "objects": {public_id: selected_history},
            "global_metric": [checkpoint.validation_metric for checkpoint in checkpoints],
            "subgroup_metrics": {
                str(subgroup_payload["subgroup_id"]): [
                    float(checkpoint.subgroup_metric) for checkpoint in checkpoints if checkpoint.subgroup_metric is not None
                ]
            },
            "subgroup_objects": {
                str(subgroup_payload["subgroup_id"]): [str(object_ids[index]) for index in validation if subgroup[index]]
            },
        }
    )
    similar_evidence = []
    selected_ids = {item.object_id: item.role for item in selected_cases}
    for item in nearest:
        role = selected_ids.get(item.reference_object_id)
        if role:
            similar_evidence.append(replace(item, is_counterexample=role == "counterexample"))
    result = FuzzyXAI.wrap(checkpoint_model, adapter="sklearn", explain_plan=plan).explain_one(
        standardized[source_index],
        object_id=public_id,
        feature_names=names,
        reference_data=standardized[train],
        reference_ids=[str(item) for item in object_ids[train]],
        reference_labels=labels[train].tolist(),
        training_run=training,
        include_training_trace=True,
        include_model_knowledge=False,
        additional_evidence=ExplanationEvidence(
            rules=(learned_rule,),
            similar_cases=tuple(similar_evidence),
            counterfactuals=tuple(sensitivity),
        ),
        dataset_version=f"bcwd_snapshot_sha256:{sha256_file(snapshot)}",
        evidence={"contributions": {name: float(value) for name, value in zip(names, checkpoint_model.coef_[0] * standardized[source_index])}, "contribution_method": "native_linear_term_x_coefficient"},
    )
    result_payload = result.to_dict()
    result_payload["trace"]["generated_at"] = "measured_reproducible_run"
    result_payload["scenario_metadata"] = {
        "scenario_id": "object_85_real_training_experiment",
        "public_case_id": public_id,
        "fixture_type": "empirical_training_run",
        "empirical_status": "measured",
        "source_type": "measured",
        "result_origin": "measured",
        "run_id": RUN_ID,
    }
    write_json(output / "explanation_result.json", result_payload)
    human = result.explain_for("domain_user")
    write_json(output / "human_explanation.json", human.to_dict(include_technical_trace=False))
    (output / "human_explanation.md").write_text(human.user_text, encoding="utf-8")

    summary = {
        "schema_version": "1.0",
        "scenario_id": "object_85_real_training_experiment",
        "run_id": RUN_ID,
        "fixture_type": "empirical_training_run",
        "empirical_status": "measured",
        "result_origin": "measured",
        "dataset": {
            "name": "Breast Cancer Wisconsin (Diagnostic)",
            "objects": len(values),
            "features": len(names),
            "license": "CC BY 4.0",
            "doi": DATASET_DOI,
            "snapshot_sha256": sha256_file(snapshot),
        },
        "split": {"train": len(train), "validation": len(validation), "test": len(test), "seed": SEED},
        "checkpoints": len(checkpoints),
        "checkpoint_hashes_unique": len({item.model_fingerprint for item in checkpoints}),
        "selected_case": {
            "public_id": public_id,
            "source_id": source_id,
            "forgetting_events": list(trace.forgetting_events),
            "rare_subgroup_member": bool(subgroup[source_index]),
        },
        "subgroup": subgroup_payload,
        "rule_ablation": ablation.to_dict(),
        "similar_case_roles": [item.role for item in selected_cases],
        "counterfactual_modes": [item.mode for item in sensitivity],
        "domain_language_validation": semantic.to_dict(),
        "cross_model_count": len(cross_model),
        "cross_model_graphs_valid": all(not row["graph_errors"] for row in cross_model),
        "comprehension_pilot": "planned_not_run",
        "release_gate": "blocked_external_pilot_and_domain_review",
        "claim_scope": "measured methodological benchmark; not clinical validation",
    }
    write_json(output / "empirical_summary.json", summary)
    environment = {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "numpy": np.__version__,
        "sklearn": __import__("sklearn").__version__,
        "seed": SEED,
        "protocol_time": PROTOCOL_TIME,
        "git_commit": subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True, capture_output=True).stdout.strip() or "unavailable",
    }
    write_json(output / "environment.json", environment)
    files = {
        str(path.relative_to(output)): sha256_file(path)
        for path in sorted(output.rglob("*"))
        if path.is_file() and path.name != "manifest_sha256.json"
    }
    write_json(
        output / "manifest_sha256.json",
        {
            "schema_version": "1.0",
            "run_id": RUN_ID,
            "source_type": "measured",
            "files": files,
        },
    )
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT))
    args = parser.parse_args()
    summary = run(args.output_dir)
    print(f"PASS: real_training checkpoints={summary['checkpoints']}")
    print(f"PASS: forgetting_case {summary['selected_case']['public_id']}")
    print("PASS: measured_rule_ablation")
    print(f"PASS: cross_model models={summary['cross_model_count']}")
    print(f"BLOCKED: comprehension_pilot {summary['comprehension_pilot']}")


if __name__ == "__main__":
    main()
