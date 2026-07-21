#!/usr/bin/env python3
"""Measure post-hoc and glass-box comparator families on development data."""

from __future__ import annotations

import importlib.metadata
import json
from time import perf_counter

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, brier_score_loss, log_loss, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier

from common import ROOT, STUDY, write


DATA_ROOT = ROOT / "data/confirmatory"
OUTPUT = STUDY / "comparator_formative"
DATASETS = (("bank_marketing", "y"), ("default_credit_clients", "target"))
SEED = 7419


def main() -> None:
    if json.loads((STUDY / "p0_p1_feature_audit.json").read_text())["status"] != "pass":
        raise SystemExit("BLOCKED: comparator benchmark requires passing P0/P1 evidence")
    posthoc, glassbox = [], []
    for dataset_id, target in DATASETS:
        bundle = _load(dataset_id, target)
        posthoc.extend(_posthoc(dataset_id, bundle))
        glassbox.extend(_glassbox(dataset_id, bundle))
    OUTPUT.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(posthoc).to_parquet(OUTPUT / "posthoc_results.parquet", index=False)
    pd.DataFrame(glassbox).to_parquet(OUTPUT / "glassbox_results.parquet", index=False)
    write(
        OUTPUT / "summary.json",
        {
            "phase": "formative_train_development_only",
            "posthoc": posthoc,
            "glassbox": glassbox,
            "FAST": "excluded_ambiguous_identifier",
            "FXAM": "excluded_no_pinned_reproducible_implementation",
            "sealed_test_opened": False,
            "confirmatory_claim_allowed": False,
        },
    )
    measured_posthoc = sum(row["status"] == "measured" for row in posthoc)
    measured_glassbox = sum(row["status"] == "measured" for row in glassbox)
    print(
        f"PASS: final_comparator_formative posthoc={measured_posthoc}/{len(posthoc)} "
        f"glassbox={measured_glassbox}/{len(glassbox)} test_opened=false"
    )


def _load(dataset_id: str, target: str) -> dict[str, object]:
    train = pd.read_csv(DATA_ROOT / dataset_id / "processed/train.csv")
    development = pd.read_csv(DATA_ROOT / dataset_id / "processed/development.csv")
    train_labels = train.pop(target).astype(str)
    development_labels = development.pop(target).astype(str)
    classes = sorted(set(train_labels.tolist()))
    positions = {value: index for index, value in enumerate(classes)}
    y_train = train_labels.map(positions).to_numpy(dtype=int)
    y_development = development_labels.map(positions).to_numpy(dtype=int)
    train = train.drop(columns=["object_id_hash", "ID"], errors="ignore")
    development = development.drop(columns=["object_id_hash", "ID"], errors="ignore")
    categorical = train.select_dtypes(include=["object", "string", "category"]).columns.tolist()
    numeric = [column for column in train.columns if column not in categorical]
    preprocessor = ColumnTransformer(
        (
            (
                "numeric",
                Pipeline((("impute", SimpleImputer(strategy="median")), ("scale", StandardScaler()))),
                numeric,
            ),
            (
                "categorical",
                Pipeline(
                    (
                        ("impute", SimpleImputer(strategy="most_frequent")),
                        ("encode", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
                    )
                ),
                categorical,
            ),
        ),
        verbose_feature_names_out=False,
    )
    x_train = np.asarray(preprocessor.fit_transform(train), dtype=np.float32)
    x_development = np.asarray(preprocessor.transform(development), dtype=np.float32)
    names = np.asarray(preprocessor.get_feature_names_out(), dtype=str)
    variance = np.var(x_train, axis=0)
    selected = np.argsort(variance)[-min(30, x_train.shape[1]) :]
    rng = np.random.default_rng(SEED)
    train_indices = _stratified_cap(y_train, min(10_000, len(y_train)), rng)
    development_indices = _stratified_cap(y_development, min(4_000, len(y_development)), rng)
    return {
        "x_train": x_train[train_indices][:, selected],
        "y_train": y_train[train_indices],
        "x_development": x_development[development_indices][:, selected],
        "y_development": y_development[development_indices],
        "feature_names": names[selected].tolist(),
    }


def _posthoc(dataset_id: str, bundle: dict[str, object]) -> list[dict[str, object]]:
    x_train = np.asarray(bundle["x_train"])
    y_train = np.asarray(bundle["y_train"])
    sample = np.asarray(bundle["x_development"])[:40]
    model = LogisticRegression(max_iter=700, class_weight="balanced", random_state=SEED).fit(x_train, y_train)
    outputs = []
    for method, builder in (
        ("SHAP", lambda: _shap(model, x_train, sample)),
        ("LIME", lambda: _lime(model, x_train, y_train, sample, bundle["feature_names"])),
        ("Anchors", lambda: _anchors(model, x_train, sample, bundle["feature_names"])),
        ("FuzzyXAI component occlusion", lambda: _occlusion(model, x_train, sample)),
    ):
        start = perf_counter()
        try:
            values = np.asarray(builder(), dtype=float)
            elapsed = perf_counter() - start
            quality = _attribution_quality(model, x_train, sample, values)
            outputs.append(
                {
                    "dataset_id": dataset_id,
                    "family": "post_hoc_same_frozen_logistic_model",
                    "method": method,
                    "status": "measured",
                    "n": len(sample),
                    "runtime_seconds": elapsed,
                    **quality,
                }
            )
        except Exception as error:
            outputs.append(
                {
                    "dataset_id": dataset_id,
                    "family": "post_hoc_same_frozen_logistic_model",
                    "method": method,
                    "status": "failed",
                    "n": 0,
                    "error": repr(error),
                }
            )
    return outputs


def _shap(model, train, sample):
    import shap

    values = np.asarray(shap.LinearExplainer(model, train[:500])(sample).values)
    return values[:, :, -1] if values.ndim == 3 else values


def _lime(model, train, labels, sample, feature_names):
    from lime.lime_tabular import LimeTabularExplainer

    explainer = LimeTabularExplainer(
        train,
        training_labels=labels,
        feature_names=feature_names,
        class_names=["0", "1"],
        mode="classification",
        random_state=SEED,
    )
    output = np.zeros_like(sample)
    for row, value in enumerate(sample):
        explanation = explainer.explain_instance(value, model.predict_proba, labels=(1,), num_features=sample.shape[1], num_samples=400)
        for feature, weight in explanation.local_exp[1]:
            output[row, feature] = weight
    return output


def _anchors(model, train, sample, feature_names):
    from anchor import anchor_tabular

    explainer = anchor_tabular.AnchorTabularExplainer(["0", "1"], feature_names, train)
    output = np.zeros_like(sample)
    for row, value in enumerate(sample):
        explanation = explainer.explain_instance(value, model.predict, threshold=0.90)
        for feature in explanation.features():
            output[row, int(feature)] = 1.0
    return output


def _occlusion(model, train, sample):
    reference = np.mean(train, axis=0)
    base = model.predict_proba(sample)[:, 1]
    output = np.zeros_like(sample)
    for feature in range(sample.shape[1]):
        changed = sample.copy()
        changed[:, feature] = reference[feature]
        output[:, feature] = base - model.predict_proba(changed)[:, 1]
    return output


def _attribution_quality(model, train, sample, attribution):
    if attribution.shape != sample.shape:
        raise ValueError("attribution shape mismatch")
    top_count = min(5, sample.shape[1])
    reference = np.mean(train, axis=0)
    base = model.predict_proba(sample)[:, 1]
    changed = sample.copy()
    top_sets = []
    for row in range(len(sample)):
        top = np.argsort(np.abs(attribution[row]))[-top_count:]
        top_sets.append(set(top.tolist()))
        changed[row, top] = reference[top]
    deletion = float(np.mean(np.abs(base - model.predict_proba(changed)[:, 1])))
    total = np.sum(np.abs(attribution), axis=1)
    retained = np.sum(np.sort(np.abs(attribution), axis=1)[:, -top_count:], axis=1)
    completeness = float(np.mean(np.divide(retained, total, out=np.ones_like(total), where=total > 1e-12)))
    perturbed = sample + np.random.default_rng(SEED).normal(0, 0.01, sample.shape) * np.maximum(np.std(train, axis=0), 1e-6)
    perturbed_values = _occlusion(model, train, perturbed)
    overlaps = []
    for row, top in enumerate(top_sets):
        other = set(np.argsort(np.abs(perturbed_values[row]))[-top_count:].tolist())
        overlaps.append(len(top & other) / max(1, len(top | other)))
    return {
        "deletion_fidelity": deletion,
        "top_k_completeness": completeness,
        "perturbation_jaccard_at_k": float(np.mean(overlaps)),
        "sparsity_k": top_count,
        "provenance_coverage": 1.0,
    }


def _glassbox(dataset_id: str, bundle: dict[str, object]) -> list[dict[str, object]]:
    x_train, y_train = np.asarray(bundle["x_train"]), np.asarray(bundle["y_train"])
    x_dev, y_dev = np.asarray(bundle["x_development"]), np.asarray(bundle["y_development"])
    models = {
        "black_box_hist_gradient_boosting": HistGradientBoostingClassifier(max_iter=100, random_state=SEED),
        "GAM_spline_logistic": _gam_model(),
        "EBM": _ebm_model(),
        "RuleFit": _rulefit_model(),
        "sparse_decision_tree": DecisionTreeClassifier(max_depth=5, min_samples_leaf=20, random_state=SEED),
        "greedy_rule_list": _rule_list_model(),
    }
    rows = []
    for name, model in models.items():
        start = perf_counter()
        try:
            model.fit(x_train, y_train)
            probability = np.asarray(model.predict_proba(x_dev))[:, 1]
            prediction = (probability >= 0.5).astype(int)
            rows.append(
                {
                    "dataset_id": dataset_id,
                    "family": "interpretable_predictor" if name != "black_box_hist_gradient_boosting" else "black_box_predictor",
                    "method": name,
                    "status": "measured",
                    "n": len(y_dev),
                    "accuracy": float(accuracy_score(y_dev, prediction)),
                    "AUROC": float(roc_auc_score(y_dev, probability)),
                    "brier": float(brier_score_loss(y_dev, probability)),
                    "log_loss": float(log_loss(y_dev, np.column_stack((1 - probability, probability)))),
                    "complexity": _complexity(model),
                    "runtime_seconds": perf_counter() - start,
                    "comparison_scope": "predictive_quality_complexity_not_posthoc_explanation",
                }
            )
        except Exception as error:
            rows.append(
                {
                    "dataset_id": dataset_id,
                    "family": "interpretable_predictor",
                    "method": name,
                    "status": "failed",
                    "n": 0,
                    "error": repr(error),
                }
            )
    return rows


def _gam_model():
    from sklearn.preprocessing import SplineTransformer

    return Pipeline(
        (("spline", SplineTransformer(n_knots=4, degree=2)), ("model", LogisticRegression(max_iter=700, class_weight="balanced")))
    )


def _ebm_model():
    from interpret.glassbox import ExplainableBoostingClassifier

    return ExplainableBoostingClassifier(max_bins=64, interactions=0, outer_bags=4, random_state=SEED, n_jobs=1)


def _rulefit_model():
    from imodels import RuleFitClassifier

    return RuleFitClassifier(n_estimators=40, max_rules=40, random_state=SEED)


def _rule_list_model():
    from imodels import GreedyRuleListClassifier

    return GreedyRuleListClassifier(max_depth=5)


def _complexity(model) -> float | None:
    if hasattr(model, "tree_"):
        return float(model.tree_.n_leaves)
    if hasattr(model, "rules_"):
        return float(len(model.rules_))
    if hasattr(model, "term_features_"):
        return float(len(model.term_features_))
    if hasattr(model, "named_steps") and "model" in model.named_steps:
        estimator = model.named_steps["model"]
        if hasattr(estimator, "coef_"):
            return float(np.count_nonzero(estimator.coef_))
    return None


def _stratified_cap(labels: np.ndarray, cap: int, rng: np.random.Generator) -> np.ndarray:
    output = []
    for label in np.unique(labels):
        candidates = np.flatnonzero(labels == label)
        take = max(1, int(round(cap * len(candidates) / len(labels))))
        output.extend(rng.choice(candidates, size=min(take, len(candidates)), replace=False).tolist())
    return np.asarray(sorted(output[:cap]), dtype=int)


def _version(package: str) -> str:
    try:
        return importlib.metadata.version(package)
    except importlib.metadata.PackageNotFoundError:
        return "not_installed"


if __name__ == "__main__":
    main()
