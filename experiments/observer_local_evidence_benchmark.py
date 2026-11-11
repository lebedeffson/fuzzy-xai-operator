"""Reproducible *structural* observer benchmark on three fitted sklearn models.

This is a developmental, seeded fault-injection benchmark. No expert labels,
natural incidents, SHAP faithfulness or cross-model generality are claimed.
The linear-logit reconstruction is an independent numerical oracle for clean
local contributions; it is not an oracle for unknown explanation methods.
"""

from __future__ import annotations

import hashlib
import json
import sys
from dataclasses import replace

import numpy as np
import sklearn
from fuzzyxai.practical_controller import (
    DeploymentContext,
    ExplanationArtifact,
    HardGuardStatus,
    PredictionArtifact,
    RouteArtifacts,
    evaluate_hard_guard,
)
from fuzzyxai.selective_observer.evidence_identity import LocalExplanationEvidence
from fuzzyxai.selective_observer.local_effects import collect_linear_reference_effects_batched, hash_numeric_input
from sklearn.datasets import load_breast_cancer, load_digits, load_wine
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


def _digest(value: object) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def _canonical_local_digest(local: LocalExplanationEvidence) -> str:
    """Hash the actual declared local record, including its contributions."""
    return _digest({
        "object_id": local.object_id,
        "model_version": local.model_version,
        "target_class": local.target_class,
        "scope": local.scope,
        "feature_names": local.feature_names,
        "contributions": local.contributions,
        "input_sha256": local.input_sha256,
        "reference_sha256": local.reference_sha256,
        "background_sha256": local.background_sha256,
        "attribution_protocol": local.attribution_protocol,
    })


def _fitted_model_sha256(model: object) -> str:
    """Hash actual fitted sklearn coefficients and scaler, not a seed label."""

    scaler = model.named_steps["standardscaler"]
    classifier = model.named_steps["logisticregression"]
    return _digest(
        {
            "estimator": "LogisticRegression",
            "scaler_mean": np.asarray(scaler.mean_, dtype=float).tolist(),
            "scaler_scale": np.asarray(scaler.scale_, dtype=float).tolist(),
            "classes": np.asarray(classifier.classes_).tolist(),
            "coefficients": np.asarray(classifier.coef_, dtype=float).tolist(),
            "intercept": np.asarray(classifier.intercept_, dtype=float).tolist(),
        }
    )


def _model_evidence(
    name: str,
    model: object,
    item: np.ndarray,
    object_id: str,
    axis: tuple[str, ...],
    seed: int,
    *,
    reference_values: np.ndarray | None = None,
):
    """Return the actual target-class linear logit decomposition."""
    scaler = model.named_steps["standardscaler"]
    logistic = model.named_steps["logisticregression"]
    values = np.asarray(scaler.transform(item.reshape(1, -1))[0], dtype=float)
    prediction = logistic.predict(values.reshape(1, -1))[0]
    classes = logistic.classes_
    target_index = int(np.where(classes == prediction)[0][0])
    if len(classes) == 2:
        direction = 1.0 if target_index == 1 else -1.0
        weights = direction * logistic.coef_[0]
        intercept = direction * logistic.intercept_[0]
        raw_logit = direction * float(logistic.decision_function(values.reshape(1, -1))[0])
    else:
        weights = logistic.coef_[target_index]
        intercept = logistic.intercept_[target_index]
        raw_logit = float(logistic.decision_function(values.reshape(1, -1))[0, target_index])
    reference = np.asarray(
        scaler.mean_ if reference_values is None else reference_values, dtype=float
    )
    reference_z = np.asarray(
        scaler.transform(reference.reshape(1, -1))[0], dtype=float
    )
    contributions = tuple(float(x) for x in weights * (values - reference_z))
    baseline_logit = float(intercept + np.dot(weights, reference_z))
    oracle_error = abs(float(baseline_logit + sum(contributions) - raw_logit))
    probabilities = tuple(float(x) for x in logistic.predict_proba(values.reshape(1, -1))[0])
    model_version = f"logreg-{name}-seed{seed}-{_fitted_model_sha256(model)[:16]}"
    input_digest = hash_numeric_input(axis, tuple(float(x) for x in item))
    reference_digest = hash_numeric_input(axis, tuple(float(x) for x in reference))
    source = {
        "object_id": object_id,
        "target_class": str(prediction),
        "model_version": model_version,
        "scope": "local",
        "input_sha256": input_digest,
        "reference_sha256": reference_digest,
        "feature_names": axis,
        "contributions": contributions,
    }
    digest = _digest(source)

    def target_logits_batch(rows: tuple[tuple[float, ...], ...]) -> tuple[float, ...]:
        model_scores = np.asarray(model.decision_function(np.asarray(rows, dtype=float)), dtype=float)
        if len(classes) == 2:
            sign = 1.0 if target_index == 1 else -1.0
            return tuple(sign * float(value) for value in model_scores)
        return tuple(float(value) for value in model_scores[:, target_index])

    effects = collect_linear_reference_effects_batched(
        object_id=object_id,
        model_version=model_version,
        target_class=str(prediction),
        feature_names=axis,
        input_values=tuple(float(x) for x in item),
        reference_values=tuple(float(x) for x in reference),
        target_logits=target_logits_batch,
    )
    prediction_artifact = PredictionArtifact(
        object_id=object_id,
        prediction=str(prediction),
        confidence=max(probabilities),
        probabilities=probabilities,
        model_version=model_version,
        feature_names=axis,
        input_sha256=input_digest,
    )
    local = LocalExplanationEvidence(
        object_id=object_id,
        model_version=model_version,
        target_class=str(prediction),
        scope="local",
        feature_names=axis,
        contributions=contributions,
        source_sha256=digest,
        input_sha256=input_digest,
        reference_sha256=reference_digest,
    )
    expl = ExplanationArtifact(
        canonical_sha256=digest,
        explainer_version="linear-logit-v1",
        model_version=model_version,
        explain_plan_version="plan-v1",
        dictionary_version="features-v1",
        available_channels=("prediction", "attribution", "provenance"),
        local_evidence=local,
    )
    route = RouteArtifacts(
        preprocessing_version="standardscaler-v1",
        calibration_version="cal-v1",
        reference_population="train-v1",
        schema_version="local-logit-v1",
        artifact_sha256=digest,
        observed_provenance_channels=("prediction", "attribution", "provenance"),
        observed_local_effects=effects,
    )
    context = DeploymentContext(
        expected_model_version=model_version,
        expected_preprocessing_version="standardscaler-v1",
        expected_explainer_version="linear-logit-v1",
        expected_calibration_version="cal-v1",
        expected_reference_population="train-v1",
        expected_schema_version="local-logit-v1",
        expected_explain_plan_version="plan-v1",
        expected_dictionary_version="features-v1",
        expected_artifact_sha256=digest,
        mandatory_provenance_channels=("prediction", "attribution", "provenance"),
        maximum_reduction_loss=0.1,
        policy_version="observer-benchmark-v1",
        require_local_evidence=True,
        require_local_effect_verification=True,
        require_input_binding=True,
        require_reference_binding=True,
    )
    return prediction_artifact, expl, route, context, oracle_error


def _faults(
    valid: LocalExplanationEvidence,
    *,
    classes: tuple[str, ...],
    rng: np.random.Generator,
    randomized_zero_sum_deltas: tuple[float, ...],
) -> dict[str, LocalExplanationEvidence]:
    axis = valid.feature_names
    contributions = valid.contributions
    assert axis is not None and contributions is not None and len(axis) >= 2
    alternate = next(candidate for candidate in classes if candidate != valid.target_class)
    swapped_axis = list(axis)
    swapped_axis[0], swapped_axis[1] = swapped_axis[1], swapped_axis[0]
    swapped_values = list(contributions)
    first = int(np.argmax(swapped_values))
    second = int(np.argmin(swapped_values))
    swapped_values[first], swapped_values[second] = swapped_values[second], swapped_values[first]
    mutations = {
        "object_substitution": replace(valid, object_id=f"{valid.object_id}-other"),
        "target_substitution": replace(valid, target_class=alternate),
        "feature_order_substitution": replace(valid, feature_names=tuple(swapped_axis)),
        "flattened_multiclass_attribution": replace(valid, contributions=contributions * len(classes)),
        "global_attribution_as_local": replace(valid, scope="global"),
        "missing_target": replace(valid, target_class=None),
        "source_digest_substitution": replace(valid, source_sha256="f" * 64),
        "stale_input_same_object": replace(valid, input_sha256="f" * 64),
        "missing_input_binding": replace(valid, input_sha256=None),
        "wrong_reference_identity": replace(valid, reference_sha256="f" * 64),
        "missing_reference_binding": replace(valid, reference_sha256=None),
        # Deliberately invisible to the structural observer: this is a
        # negative control that motivates independent faithfulness checks.
        "silent_value_permutation": replace(valid, contributions=tuple(swapped_values)),
    }
    for delta in randomized_zero_sum_deltas:
        left, right = rng.choice(len(contributions), size=2, replace=False)
        changed = list(contributions)
        changed[int(left)] += delta
        changed[int(right)] -= delta
        mutations[f"randomized_zero_sum_{delta:g}"] = replace(valid, contributions=tuple(changed))
    return mutations


def run_benchmark(
    *,
    n_per_dataset: int = 12,
    seed: int = 42,
    randomized_zero_sum_deltas: tuple[float, ...] = (1e-4, 1e-2, 0.1),
) -> dict[str, object]:
    """Train on disjoint data and evaluate clean vs corrupted *test* artifacts."""
    if n_per_dataset <= 0:
        raise ValueError("n_per_dataset must be positive")
    if any(not np.isfinite(delta) or delta <= 1e-6 for delta in randomized_zero_sum_deltas):
        raise ValueError("randomized deltas must exceed the fixed verification tolerance")
    datasets = {
        "breast_cancer": load_breast_cancer(),
        "wine": load_wine(),
        "digits": load_digits(),
    }
    rows: list[dict[str, object]] = []
    model_fingerprints: dict[str, str] = {}
    dataset_metadata: dict[str, dict[str, int]] = {}
    rng = np.random.default_rng(seed + 8192)
    largest_oracle_residual = 0.0
    for dataset_name, ds in datasets.items():
        features = np.asarray(ds.data, dtype=float)
        labels = np.asarray(ds.target)
        train, test = train_test_split(
            np.arange(len(labels)), test_size=0.25, random_state=seed, stratify=labels
        )
        model = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1500, random_state=seed))
        model.fit(features[train], labels[train])
        model_fingerprints[dataset_name] = _fitted_model_sha256(model)
        dataset_metadata[dataset_name] = {
            "n_train": len(train),
            "n_test": len(test),
            "n_features": features.shape[1],
            "n_classes": len(model.named_steps["logisticregression"].classes_),
        }
        axis = tuple(str(s) for s in getattr(ds, "feature_names", [f"f{i}" for i in range(features.shape[1])]))
        classes = tuple(str(label) for label in model.named_steps["logisticregression"].classes_)
        if n_per_dataset > len(test):
            raise ValueError("n_per_dataset exceeds size of test partition")
        for index in test[:n_per_dataset]:
            prediction, explanation, route, context, oracle_error = _model_evidence(
                dataset_name, model, features[index], f"{dataset_name}-{index}", axis, seed
            )
            observed_effects = route.observed_local_effects
            assert observed_effects is not None and observed_effects.effect_on_score is not None
            assert explanation.local_evidence is not None and explanation.local_evidence.contributions is not None
            max_effect_residual = max(
                abs(float(x) - float(y))
                for x, y in zip(explanation.local_evidence.contributions, observed_effects.effect_on_score, strict=True)
            )
            largest_oracle_residual = max(largest_oracle_residual, oracle_error, max_effect_residual)
            valid = explanation.local_evidence
            assert valid is not None
            variants = {
                "clean": valid,
                **_faults(
                    valid,
                    classes=classes,
                    rng=rng,
                    randomized_zero_sum_deltas=randomized_zero_sum_deltas,
                ),
            }
            for fault_family, local in variants.items():
                if fault_family != "source_digest_substitution":
                    # Simulate a consistently (re)hashed, but semantically
                    # WRONG explainer output rather than in-transit corruption.
                    # This makes a SHA256 integrity check an honest baseline.
                    digest = _canonical_local_digest(local)
                    local = replace(local, source_sha256=digest)
                    variant_explanation = replace(
                        explanation, canonical_sha256=digest, local_evidence=local
                    )
                    variant_route = replace(route, artifact_sha256=digest)
                    variant_context = replace(context, expected_artifact_sha256=digest)
                else:
                    variant_explanation = replace(explanation, local_evidence=local)
                    variant_route = route
                    variant_context = context
                observed = evaluate_hard_guard(
                    prediction, variant_explanation, variant_route, variant_context
                )
                structural_only = evaluate_hard_guard(
                    prediction,
                    variant_explanation,
                    replace(variant_route, observed_local_effects=None),
                    replace(
                        variant_context,
                        require_local_effect_verification=False,
                        require_reference_binding=False,
                    ),
                )
                verified_hash = (
                    local.source_sha256 == variant_explanation.canonical_sha256
                    and variant_explanation.canonical_sha256 == _canonical_local_digest(local)
                )
                sum_only_accepted = (
                    structural_only.status is HardGuardStatus.CERTIFIED
                    and local.contributions is not None
                    and observed_effects.reference_score is not None
                    and observed_effects.explained_score is not None
                    and abs(
                        sum(local.contributions)
                        - (observed_effects.explained_score - observed_effects.reference_score)
                    ) <= 1e-6
                )
                identity_baseline_accepts = (
                    local.object_id == prediction.object_id
                    and local.model_version == prediction.model_version
                    and verified_hash
                )
                rows.append(
                    {
                        "dataset": dataset_name,
                        "fault_family": fault_family,
                        "object_id": prediction.object_id,
                        "status": observed.status.value,
                        "structural_only_status": structural_only.status.value,
                        "baseline_accepts": identity_baseline_accepts,
                        "verified_hash": verified_hash,
                        "structural_plus_sum_accepted": sum_only_accepted,
                    }
                )

    def summarize(group: list[dict[str, object]]) -> dict[str, int]:
        return {
            "n": len(group),
            "blocked": sum(row["status"] == HardGuardStatus.BLOCKED.value for row in group),
            "review_required": sum(row["status"] == HardGuardStatus.REVIEW_REQUIRED.value for row in group),
            "accepted_structurally": sum(row["status"] == HardGuardStatus.CERTIFIED.value for row in group),
            "identity_baseline_accepted": sum(bool(row["baseline_accepts"]) for row in group),
            "hash_verified": sum(bool(row["verified_hash"]) for row in group),
            "structural_only_accepted": sum(row["structural_only_status"] == HardGuardStatus.CERTIFIED.value for row in group),
            "structural_plus_sum_accepted": sum(bool(row["structural_plus_sum_accepted"]) for row in group),
        }

    families = sorted({str(row["fault_family"]) for row in rows})
    grouped = {
        family: summarize([row for row in rows if row["fault_family"] == family])
        for family in families
    }
    by_dataset = {
        name: {
            family: summarize([
                row for row in rows if row["dataset"] == name and row["fault_family"] == family
            ])
            for family in families
        }
        for name in datasets
    }

    return {
        "protocol": "three sklearn datasets / stratified train-test / seeded corruptions",
        "phase": "developmental_fault_injection",
        "n_per_dataset": n_per_dataset,
        "seed": seed,
        "randomized_zero_sum_deltas": randomized_zero_sum_deltas,
        "n_rows": len(rows),
        "maximum_clean_logit_reconstruction_error": largest_oracle_residual,
        "by_family": grouped,
        "by_dataset": by_dataset,
        "model_fingerprints": model_fingerprints,
        "dataset_metadata": dataset_metadata,
        "environment": {"python": sys.version.split()[0], "numpy": np.__version__, "sklearn": sklearn.__version__},
        "model_probe_cost_per_example": {
            name: {"source_model_calls": 1, "model_score_rows": len(ds.feature_names) + 2}
            for name, ds in datasets.items()
        },
        "interpretation_limit": "developmental seeded fault injection; per-object cases correlated; canonical hashes are correctly recomputed for erroneous outputs, but provenance trust is not independently established; nonlinear and interacting explainers unsupported",
        "claim_allowed": False,
    }


if __name__ == "__main__":
    print(json.dumps(run_benchmark(), ensure_ascii=False, indent=2, sort_keys=True))
