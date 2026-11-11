"""Developmental comparison of native SHAP and an independent exact score game.

These are developer-generated, *correlated* fault variants of held-out
classification objects. The experiment does not establish unknown-fault
recall, explainability utility, causal effects or generalization to other
background games or XAI algorithms. No protected confirmatory data are used.
"""

from __future__ import annotations

import hashlib
import json
import pickle
import sys
from dataclasses import replace

import numpy as np
import shap
import sklearn
from fuzzyxai.practical_controller import (
    DeploymentContext,
    ExplanationArtifact,
    HardGuardStatus,
    PredictionArtifact,
    RouteArtifacts,
    evaluate_hard_guard,
)
from fuzzyxai.selective_observer import (
    LocalExplanationEvidence,
    collect_exact_interventional_shapley,
)
from sklearn.datasets import load_breast_cancer, load_digits, load_wine
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier

from experiments.observer_local_evidence_benchmark import _canonical_local_digest


def _datasets() -> dict[str, tuple[object, tuple[int, ...], object]]:
    return {
        "breast_cancer": (
            load_breast_cancer(), (0, 1, 4, 6),
            DecisionTreeClassifier(max_depth=5, random_state=2026),
        ),
        "wine": (
            load_wine(), (0, 2, 5, 9),
            RandomForestClassifier(n_estimators=16, max_depth=5, random_state=2026),
        ),
        "digits": (
            load_digits(), (20, 27, 35, 44),
            RandomForestClassifier(n_estimators=16, max_depth=6, random_state=2026),
        ),
    }


def _make_artifacts(
    object_id: str,
    version: str,
    names: tuple[str, ...],
    probabilities: tuple[float, ...],
    target_label: str,
    contributions: tuple[float, ...],
    oracle,
) -> tuple[PredictionArtifact, ExplanationArtifact, RouteArtifacts, DeploymentContext]:
    local = LocalExplanationEvidence(
        object_id=object_id,
        model_version=version,
        target_class=target_label,
        scope="local",
        attribution_protocol="exact_empirical_interventional_shapley_v1",
        feature_names=names,
        contributions=contributions,
        input_sha256=oracle.input_sha256,
        background_sha256=oracle.background_sha256,
    )
    digest = _canonical_local_digest(local)
    local = replace(local, source_sha256=digest)
    prediction = PredictionArtifact(
        object_id=object_id,
        prediction=target_label,
        confidence=max(probabilities),
        probabilities=probabilities,
        model_version=version,
        feature_names=names,
        input_sha256=oracle.input_sha256,
    )
    explanation = ExplanationArtifact(
        canonical_sha256=digest,
        explainer_version="shap-exact-independent-masker-v1",
        model_version=version,
        explain_plan_version="plan-exact-v1",
        dictionary_version="features-v1",
        available_channels=("prediction", "attribution", "provenance"),
        local_evidence=local,
    )
    route = RouteArtifacts(
        preprocessing_version="selected-raw-features-v1",
        calibration_version="calibration-not-claimed-v1",
        reference_population="train-subset-8-v1",
        schema_version="exact-shapley-4d-v1",
        artifact_sha256=digest,
        observed_provenance_channels=("prediction", "attribution", "provenance"),
        observed_exact_shapley=oracle,
    )
    context = DeploymentContext(
        expected_model_version=version,
        expected_preprocessing_version=route.preprocessing_version,
        expected_explainer_version=explanation.explainer_version,
        expected_calibration_version=route.calibration_version,
        expected_reference_population=route.reference_population,
        expected_schema_version=route.schema_version,
        expected_explain_plan_version=explanation.explain_plan_version,
        expected_dictionary_version=explanation.dictionary_version,
        expected_artifact_sha256=digest,
        mandatory_provenance_channels=route.observed_provenance_channels,
        maximum_reduction_loss=0.1,
        policy_version="exact-shapley-benchmark-v1",
        require_local_evidence=True,
        require_input_binding=True,
        require_exact_shapley_verification=True,
    )
    return prediction, explanation, route, context


def run_benchmark(*, n_per_dataset: int = 12, seed: int = 2026) -> dict[str, object]:
    """Never tune a gate on a tested object: this is a descriptive probe."""
    if n_per_dataset <= 0:
        raise ValueError("n_per_dataset must be positive")
    rows: list[dict[str, object]] = []
    models: dict[str, object] = {}
    maximum_clean_disagreement = 0.0
    for name, (ds, indices, estimator) in _datasets().items():
        features = np.asarray(ds.data, dtype=float)[:, list(indices)]
        labels = np.asarray(ds.target)
        train, test = train_test_split(
            np.arange(len(labels)), test_size=0.25, random_state=seed, stratify=labels
        )
        estimator.fit(features[train], labels[train])
        if n_per_dataset > len(test):
            raise ValueError("n_per_dataset exceeds held-out partition")
        background = np.asarray(features[train[:8]], dtype=float)
        mask = shap.maskers.Independent(background, max_samples=8)
        names = tuple(str(ds.feature_names[j]) for j in indices)
        fingerprint = hashlib.sha256(pickle.dumps(estimator, protocol=4)).hexdigest()
        model_version = f"exact-{name}-{fingerprint[:16]}"
        models[name] = {
            "sha256_pickle_state": fingerprint,
            "n_train": len(train),
            "n_test": len(test),
            "n_features": len(names),
            "background_rows": 8,
            "n_classes": len(estimator.classes_),
            "estimator": type(estimator).__name__,
        }
        explainers: dict[int, object] = {}
        for sample_index in test[:n_per_dataset]:
            item = features[sample_index]
            probabilities = tuple(
                float(v) for v in estimator.predict_proba(item.reshape(1, -1))[0]
            )
            target_index = int(np.argmax(probabilities))
            target_label = str(estimator.classes_[target_index])

            def target_model(matrix, chosen=target_index, fitted=estimator):
                return fitted.predict_proba(np.asarray(matrix, dtype=float))[:, chosen]

            def scorer(batch):
                return tuple(
                    float(v) for v in target_model(np.asarray(batch, dtype=float))
                )

            if target_index not in explainers:
                explainers[target_index] = shap.ExactExplainer(target_model, mask)
            native = np.asarray(
                explainers[target_index](item.reshape(1, -1)).values[0], dtype=float
            )
            oracle = collect_exact_interventional_shapley(
                object_id=f"{name}-{sample_index}",
                model_version=model_version,
                target_class=target_label,
                feature_names=names,
                input_values=tuple(float(v) for v in item),
                background_rows=tuple(
                    tuple(float(v) for v in row) for row in background
                ),
                target_scores=scorer,
            )
            assert oracle.attributions is not None
            clean_error = max(
                abs(float(a) - float(b))
                for a, b in zip(native, oracle.attributions, strict=True)
            )
            maximum_clean_disagreement = max(maximum_clean_disagreement, clean_error)
            predicted, original, source_route, source_context = _make_artifacts(
                oracle.object_id, model_version, names, probabilities, target_label,
                tuple(float(v) for v in native), oracle,
            )
            assert original.local_evidence is not None
            local = original.local_evidence
            assert local.contributions is not None
            faults = {"clean": local}
            for magnitude in (0.02, 0.2):
                corrupted = list(local.contributions)
                corrupted[0] += magnitude
                corrupted[1] -= magnitude
                faults[f"zero_sum_{magnitude:g}"] = replace(
                    local, contributions=tuple(corrupted)
                )
            faults["wrong_background"] = replace(
                local, background_sha256="f" * 64
            )
            faults["missing_background"] = replace(
                local, background_sha256=None
            )
            faults["stale_input"] = replace(local, input_sha256="f" * 64)

            for family, variant in faults.items():
                digest = _canonical_local_digest(variant)
                variant = replace(variant, source_sha256=digest)
                explanation = replace(
                    original, local_evidence=variant, canonical_sha256=digest
                )
                route = replace(source_route, artifact_sha256=digest)
                context = replace(source_context, expected_artifact_sha256=digest)
                result = evaluate_hard_guard(
                    predicted, explanation, route, context
                )
                structural_baseline = evaluate_hard_guard(
                    predicted,
                    explanation,
                    replace(route, observed_exact_shapley=None),
                    replace(context, require_exact_shapley_verification=False),
                )
                strong_sum_baseline = (
                    structural_baseline.status is HardGuardStatus.CERTIFIED
                    and abs(sum(variant.contributions) - (
                        oracle.explained_score - oracle.baseline_score
                    )) <= 1e-6
                )
                rows.append({
                    "dataset": name,
                    "object_id": oracle.object_id,
                    "family": family,
                    "status": result.status.value,
                    "structural_status": structural_baseline.status.value,
                    "structure_plus_sum_accepts": strong_sum_baseline,
                })
    names = sorted({row["family"] for row in rows})
    counts = {
        name: {
            "n": sum(r["family"] == name for r in rows),
            "certified": sum(
                r["family"] == name and r["status"] == HardGuardStatus.CERTIFIED.value
                for r in rows
            ),
            "blocked": sum(
                r["family"] == name and r["status"] == HardGuardStatus.BLOCKED.value
                for r in rows
            ),
            "review": sum(
                r["family"] == name and r["status"] == HardGuardStatus.REVIEW_REQUIRED.value
                for r in rows
            ),
            "strong_sum_baseline_accepted": sum(
                r["family"] == name and bool(r["structure_plus_sum_accepts"])
                for r in rows
            ),
        }
        for name in names
    }
    return {
        "phase": "developmental_nonlinear_exact_shapley",
        "protocol": "empirical marginal replacement score game; native SHAP ExactExplainer",
        "n_heldout_objects": len(rows) // len(names),
        "n_fault_variants": len(rows),
        "seed": seed,
        "per_dataset_requested": n_per_dataset,
        "maximum_clean_native_vs_oracle_disagreement": maximum_clean_disagreement,
        "families": counts,
        "datasets": models,
        "environment": {
            "python": sys.version.split()[0],
            "numpy": np.__version__,
            "sklearn": sklearn.__version__,
            "shap": shap.__version__,
        },
        "limitations": [
            "Only a fixed 4-feature empirical interventional Shapley game",
            "Native SHAP ExactExplainer uses the same theoretical game as the oracle",
            "Faults are designer-created and correlated within each held-out object",
            "Tree models and selected columns reuse three known sklearn datasets",
            "No independent unknown-fault experiment, real users, causal or model-quality claims",
        ],
        "claim_allowed": False,
    }


if __name__ == "__main__":
    print(json.dumps(run_benchmark(), indent=2, ensure_ascii=False, sort_keys=True))
