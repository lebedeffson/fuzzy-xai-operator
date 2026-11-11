"""Reproducible *structural* observer benchmark on three fitted sklearn models.

This is a developmental, seeded fault-injection benchmark. No expert labels,
natural incidents, SHAP faithfulness or cross-model generality are claimed.
The linear-logit reconstruction is an independent numerical oracle for clean
local contributions; it is not an oracle for unknown explanation methods.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, replace

import numpy as np
from fuzzyxai.practical_controller import (
    DeploymentContext,
    ExplanationArtifact,
    HardGuardStatus,
    PredictionArtifact,
    RouteArtifacts,
    evaluate_hard_guard,
)
from fuzzyxai.selective_observer.evidence_identity import LocalExplanationEvidence
from fuzzyxai.selective_observer.local_effects import LocalEffectEvidence, SUPPORTED_PROTOCOL
from sklearn.datasets import load_breast_cancer, load_digits, load_wine
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


def _digest(value: object) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def _model_evidence(name: str, model: object, item: np.ndarray, object_id: str, axis: tuple[str, ...]):
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
    contributions = tuple(float(x) for x in weights * values)
    oracle_error = abs(float(intercept + sum(contributions) - raw_logit))
    probabilities = tuple(float(x) for x in logistic.predict_proba(values.reshape(1, -1))[0])
    model_version = f"logreg-{name}-seed42"
    source = {
        "object_id": object_id,
        "target": str(prediction),
        "model_version": model_version,
        "feature_names": axis,
        "contributions": contributions,
        "intercept": float(intercept),
    }
    digest = _digest(source)

    def target_logit(raw_values: np.ndarray) -> float:
        model_scores = np.asarray(model.decision_function(raw_values.reshape(1, -1)), dtype=float)
        if len(classes) == 2:
            return (1.0 if target_index == 1 else -1.0) * float(model_scores[0])
        return float(model_scores[0, target_index])

    original_logit = target_logit(item)
    reference = np.asarray(scaler.mean_, dtype=float)
    reference_logit = target_logit(reference)
    observed_effects = []
    for feature_index in range(len(axis)):
        counterfactual = item.copy()
        counterfactual[feature_index] = reference[feature_index]
        observed_effects.append(original_logit - target_logit(counterfactual))
    effects = LocalEffectEvidence(
        protocol=SUPPORTED_PROTOCOL,
        object_id=object_id,
        model_version=model_version,
        target_class=str(prediction),
        feature_names=axis,
        effect_on_score=tuple(float(x) for x in observed_effects),
        reference_score=reference_logit,
        explained_score=original_logit,
    )
    prediction_artifact = PredictionArtifact(
        object_id=object_id,
        prediction=str(prediction),
        confidence=max(probabilities),
        probabilities=probabilities,
        model_version=model_version,
        feature_names=axis,
    )
    local = LocalExplanationEvidence(
        object_id=object_id,
        model_version=model_version,
        target_class=str(prediction),
        scope="local",
        feature_names=axis,
        contributions=contributions,
        source_sha256=digest,
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
    )
    return prediction_artifact, expl, route, context, oracle_error


def _faults(valid: LocalExplanationEvidence, *, classes: tuple[str, ...]):
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
    return {
        "object_substitution": replace(valid, object_id=f"{valid.object_id}-other"),
        "target_substitution": replace(valid, target_class=alternate),
        "feature_order_substitution": replace(valid, feature_names=tuple(swapped_axis)),
        "flattened_multiclass_attribution": replace(valid, contributions=contributions * len(classes)),
        "global_attribution_as_local": replace(valid, scope="global"),
        "missing_target": replace(valid, target_class=None),
        "source_digest_substitution": replace(valid, source_sha256="f" * 64),
        # Deliberately invisible to the structural observer: this is a
        # negative control that motivates independent faithfulness checks.
        "silent_value_permutation": replace(valid, contributions=tuple(swapped_values)),
    }


def run_benchmark(*, n_per_dataset: int = 12, seed: int = 42) -> dict[str, object]:
    """Train on disjoint data and evaluate clean vs corrupted *test* artifacts."""
    if n_per_dataset <= 0:
        raise ValueError("n_per_dataset must be positive")
    datasets = {
        "breast_cancer": load_breast_cancer(),
        "wine": load_wine(),
        "digits": load_digits(),
    }
    rows: list[dict[str, object]] = []
    largest_oracle_residual = 0.0
    for dataset_name, ds in datasets.items():
        features = np.asarray(ds.data, dtype=float)
        labels = np.asarray(ds.target)
        train, test = train_test_split(
            np.arange(len(labels)), test_size=0.25, random_state=seed, stratify=labels
        )
        model = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1500, random_state=seed))
        model.fit(features[train], labels[train])
        axis = tuple(str(s) for s in getattr(ds, "feature_names", [f"f{i}" for i in range(features.shape[1])]))
        classes = tuple(str(label) for label in model.named_steps["logisticregression"].classes_)
        if n_per_dataset > len(test):
            raise ValueError("n_per_dataset exceeds size of test partition")
        for index in test[:n_per_dataset]:
            prediction, explanation, route, context, oracle_error = _model_evidence(
                dataset_name, model, features[index], f"{dataset_name}-{index}", axis
            )
            largest_oracle_residual = max(largest_oracle_residual, oracle_error)
            valid = explanation.local_evidence
            assert valid is not None
            variants = {"clean": valid, **_faults(valid, classes=classes)}
            for fault_family, local in variants.items():
                observed = evaluate_hard_guard(
                    prediction, replace(explanation, local_evidence=local), route, context
                )
                structural_only = evaluate_hard_guard(
                    prediction,
                    replace(explanation, local_evidence=local),
                    replace(route, observed_local_effects=None),
                    replace(context, require_local_effect_verification=False),
                )
                identity_baseline_accepts = (
                    local.object_id == prediction.object_id
                    and local.model_version == prediction.model_version
                    and local.source_sha256 == explanation.canonical_sha256
                )
                rows.append(
                    {
                        "dataset": dataset_name,
                        "fault_family": fault_family,
                        "object_id": prediction.object_id,
                        "status": observed.status.value,
                        "structural_only_status": structural_only.status.value,
                        "baseline_accepts": identity_baseline_accepts,
                    }
                )

    grouped = {}
    for family in sorted({str(row["fault_family"]) for row in rows}):
        group = [row for row in rows if row["fault_family"] == family]
        grouped[family] = {
            "n": len(group),
            "blocked": sum(row["status"] == HardGuardStatus.BLOCKED.value for row in group),
            "review_required": sum(row["status"] == HardGuardStatus.REVIEW_REQUIRED.value for row in group),
            "accepted_structurally": sum(row["status"] == HardGuardStatus.CERTIFIED.value for row in group),
            "identity_baseline_accepted": sum(bool(row["baseline_accepts"]) for row in group),
            "structural_only_accepted": sum(row["structural_only_status"] == HardGuardStatus.CERTIFIED.value for row in group),
        }

    return {
        "protocol": "three sklearn datasets / stratified train-test / seeded corruptions",
        "phase": "developmental_fault_injection",
        "n_per_dataset": n_per_dataset,
        "seed": seed,
        "n_rows": len(rows),
        "maximum_clean_logit_reconstruction_error": largest_oracle_residual,
        "by_family": grouped,
        "interpretation_limit": "independent linear-logit model-effect checks only; interactions and arbitrary nonlinear explainers unsupported",
        "claim_allowed": False,
    }


if __name__ == "__main__":
    print(json.dumps(run_benchmark(), ensure_ascii=False, indent=2, sort_keys=True))
