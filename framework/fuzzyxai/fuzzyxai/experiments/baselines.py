"""Real optional-baseline invocations for E3.

Methods are never substituted silently. If a library cannot be imported or a
method fails, its row is marked ``failed`` and the full evidence gate remains
blocked.
"""

from __future__ import annotations

import importlib.metadata
from dataclasses import asdict, dataclass
from time import perf_counter
from typing import Any, Callable, Sequence

import numpy as np


@dataclass(frozen=True)
class BaselineResult:
    method: str
    status: str
    implementation: str
    version: str | None
    n_explained: int
    elapsed_seconds: float | None
    fidelity: float | None
    stability: float | None
    completeness: float | None
    sparsity: float | None
    traceability: float
    model_calls: int | None
    limitation: str | None = None

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def run_optional_baselines(
    *,
    model: Any,
    train_values: np.ndarray,
    train_labels: np.ndarray,
    test_values: np.ndarray,
    test_labels: np.ndarray,
    feature_names: Sequence[str],
    sample_size: int,
    seed: int = 42,
) -> list[BaselineResult]:
    sample = np.asarray(test_values[:sample_size], dtype=float)
    labels = np.asarray(test_labels[:sample_size], dtype=int)
    return [
        _run_attribution("SHAP", "shap", lambda: _shap_values(model, train_values, sample), model, train_values, sample),
        _run_attribution(
            "LIME",
            "lime",
            lambda: _lime_values(model, train_values, sample, feature_names, seed),
            model,
            train_values,
            sample,
        ),
        _run_anchor(model, train_values, sample, labels, feature_names, seed),
        _run_rulefit(train_values, train_labels, test_values, test_labels, feature_names, seed),
    ]


def _run_attribution(
    method: str,
    package: str,
    build: Callable[[], np.ndarray],
    model: Any,
    background: np.ndarray,
    sample: np.ndarray,
) -> BaselineResult:
    try:
        version = importlib.metadata.version(package)
        start = perf_counter()
        attribution = np.asarray(build(), dtype=float)
        elapsed = perf_counter() - start
        quality = attribution_quality(model, background, sample, attribution)
        return BaselineResult(
            method=method,
            status="measured",
            implementation=package,
            version=version,
            n_explained=len(sample),
            elapsed_seconds=elapsed,
            fidelity=quality["fidelity"],
            stability=quality["stability"],
            completeness=quality["completeness"],
            sparsity=quality["sparsity"],
            traceability=1.0,
            model_calls=quality["model_calls"],
        )
    except Exception as error:  # optional methods must fail closed in evidence.
        return BaselineResult(method, "failed", package, _version_or_none(package), 0, None, None, None, None, None, 0.0, None, repr(error))


def _shap_values(model: Any, train_values: np.ndarray, sample: np.ndarray) -> np.ndarray:
    import shap

    background = np.asarray(train_values[: min(200, len(train_values))], dtype=float)
    explainer = shap.Explainer(model, background)
    values = np.asarray(explainer(sample).values)
    if values.ndim == 3:
        values = values[:, :, -1]
    return values


def _lime_values(
    model: Any,
    train_values: np.ndarray,
    sample: np.ndarray,
    feature_names: Sequence[str],
    seed: int,
) -> np.ndarray:
    from lime.lime_tabular import LimeTabularExplainer

    explainer = LimeTabularExplainer(
        np.asarray(train_values, dtype=float),
        feature_names=list(feature_names),
        class_names=["0", "1"],
        mode="classification",
        random_state=seed,
    )
    values = np.zeros_like(sample, dtype=float)
    for row_index, row in enumerate(sample):
        explanation = explainer.explain_instance(row, model.predict_proba, num_features=sample.shape[1], num_samples=600)
        for feature_index, weight in explanation.local_exp[1]:
            values[row_index, feature_index] = float(weight)
    return values


def _run_anchor(
    model: Any,
    train_values: np.ndarray,
    sample: np.ndarray,
    labels: np.ndarray,
    feature_names: Sequence[str],
    seed: int,
) -> BaselineResult:
    package = "alibi"
    try:
        version = importlib.metadata.version(package)
        from alibi.explainers import AnchorTabular

        start = perf_counter()
        explainer = AnchorTabular(model.predict, list(feature_names), seed=seed)
        explainer.fit(np.asarray(train_values, dtype=float), disc_perc=(25, 50, 75))
        explanations = [explainer.explain(row, threshold=0.9) for row in sample]
        elapsed = perf_counter() - start
        precisions = [float(item.data.get("precision", 0.0)) for item in explanations]
        anchor_sizes = [len(item.data.get("anchor", ())) for item in explanations]
        return BaselineResult(
            "Anchors",
            "measured",
            package,
            version,
            len(sample),
            elapsed,
            float(np.mean(precisions)),
            None,
            float(np.mean(precisions)),
            float(np.mean(anchor_sizes)),
            1.0,
            None,
        )
    except Exception as error:
        return BaselineResult("Anchors", "failed", package, _version_or_none(package), 0, None, None, None, None, None, 0.0, None, repr(error))


def _run_rulefit(
    train_values: np.ndarray,
    train_labels: np.ndarray,
    test_values: np.ndarray,
    test_labels: np.ndarray,
    feature_names: Sequence[str],
    seed: int,
) -> BaselineResult:
    package = "imodels"
    try:
        version = importlib.metadata.version(package)
        from imodels import RuleFitClassifier

        start = perf_counter()
        model = RuleFitClassifier(random_state=seed, max_rules=30)
        model.fit(np.asarray(train_values), np.asarray(train_labels), feature_names=list(feature_names))
        predictions = np.asarray(model.predict(np.asarray(test_values)), dtype=int)
        elapsed = perf_counter() - start
        fidelity = float(np.mean(predictions == np.asarray(test_labels)))
        rules = model._get_rules(exclude_zero_coef=True)
        return BaselineResult("RuleFit", "measured", package, version, len(test_values), elapsed, fidelity, None, fidelity, float(len(rules)), 1.0, None)
    except Exception as error:
        return BaselineResult("RuleFit", "failed", package, _version_or_none(package), 0, None, None, None, None, None, 0.0, None, repr(error))


def attribution_quality(
    model: Any,
    background: np.ndarray,
    sample: np.ndarray,
    attribution: np.ndarray,
) -> dict[str, float | int]:
    if attribution.shape != sample.shape:
        raise ValueError(f"attribution shape {attribution.shape} does not match sample {sample.shape}")
    reference = np.mean(np.asarray(background, dtype=float), axis=0)
    base_probability = np.asarray(model.predict_proba(sample), dtype=float)[:, 1]
    top_count = max(1, min(5, sample.shape[1]))
    deleted = sample.copy()
    for row_index in range(len(sample)):
        top = np.argsort(np.abs(attribution[row_index]))[-top_count:]
        deleted[row_index, top] = reference[top]
    deleted_probability = np.asarray(model.predict_proba(deleted), dtype=float)[:, 1]
    deletion = np.mean(np.abs(base_probability - deleted_probability))
    absolute = np.abs(attribution)
    completeness = float(np.mean(np.sum(np.sort(absolute, axis=1)[:, -top_count:], axis=1) / np.maximum(np.sum(absolute, axis=1), 1e-12)))
    rng = np.random.default_rng(901)
    perturbed = sample + rng.normal(0.0, 0.01 * np.maximum(np.std(background, axis=0), 1e-9), size=sample.shape)
    prediction_stability = 1.0 - float(np.mean(np.abs(np.asarray(model.predict_proba(perturbed))[:, 1] - base_probability)))
    return {
        "fidelity": float(deletion),
        "stability": max(0.0, prediction_stability),
        "completeness": completeness,
        "sparsity": float(top_count),
        "model_calls": int(2 * len(sample)),
    }


def _version_or_none(package: str) -> str | None:
    try:
        return importlib.metadata.version(package)
    except importlib.metadata.PackageNotFoundError:
        return None
