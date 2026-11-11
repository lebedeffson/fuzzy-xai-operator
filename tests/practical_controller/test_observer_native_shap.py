"""Native SHAP LinearExplainer compatibility and adversarial value checks.

The source explanation is genuinely produced by SHAP. This test only
establishes agreement for binary logistic target logits and the exact
training-background/linear-additive reference used here. SHAP/LIME
generality, arbitrary background distributions, causality and user
comprehension are not asserted.
"""

from __future__ import annotations

from dataclasses import replace

import numpy as np
import shap
from experiments.observer_local_evidence_benchmark import _canonical_local_digest, _model_evidence
from fuzzyxai.practical_controller import HardGuardStatus, evaluate_hard_guard
from sklearn.datasets import load_breast_cancer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


def test_native_shap_linear_explainer_is_accepted_and_zero_sum_fault_blocked() -> None:
    dataset = load_breast_cancer()
    X = np.asarray(dataset.data, dtype=float)
    y = np.asarray(dataset.target, dtype=int)
    training, heldout = train_test_split(
        np.arange(len(y)), test_size=0.25, stratify=y, random_state=42
    )
    pipeline = make_pipeline(
        StandardScaler(), LogisticRegression(max_iter=1500, random_state=42)
    )
    pipeline.fit(X[training], y[training])
    scaler = pipeline.named_steps["standardscaler"]
    model = pipeline.named_steps["logisticregression"]
    background = scaler.transform(X[training])
    candidates = scaler.transform(X[heldout])
    positive = np.where(model.predict(candidates) == 1)[0][:3]
    assert len(positive) == 3
    samples = candidates[positive]

    # SHAP's default Independent masker samples 100 background rows:
    # the source-model probe must use exactly THAT reference mean.
    masker = shap.maskers.Independent(background)
    native_explainer = shap.LinearExplainer(model, masker)
    native = np.asarray(native_explainer(samples).values, dtype=float)
    reference_z = np.asarray(masker.data, dtype=float).mean(axis=0)
    reference_raw = scaler.inverse_transform(reference_z.reshape(1, -1))[0]
    assert native.shape == (3, X.shape[1])
    assert np.isfinite(native).all()
    feature_axis = tuple(str(s) for s in dataset.feature_names)

    for source_row, attributions in zip(heldout[positive], native, strict=True):
        prediction, explanation, route, context, _ = _model_evidence(
            "breast_cancer_native_shap",
            pipeline,
            X[source_row],
            f"shap-{source_row}",
            feature_axis,
            42,
            reference_values=reference_raw,
        )
        assert prediction.prediction == "1"
        original = explanation.local_evidence
        assert original is not None
        clean_local = replace(original, contributions=tuple(float(v) for v in attributions))
        digest = _canonical_local_digest(clean_local)
        clean_local = replace(clean_local, source_sha256=digest)
        clean_explanation = replace(
            explanation, local_evidence=clean_local, canonical_sha256=digest
        )
        clean_route = replace(route, artifact_sha256=digest)
        clean_context = replace(context, expected_artifact_sha256=digest)
        accepted = evaluate_hard_guard(
            prediction, clean_explanation, clean_route, clean_context
        )
        assert accepted.status is HardGuardStatus.CERTIFIED, accepted.reason_codes

        swapped = list(clean_local.contributions)
        swapped[0] += 0.01
        swapped[1] -= 0.01
        corrupted_local = replace(clean_local, contributions=tuple(swapped))
        new_digest = _canonical_local_digest(corrupted_local)
        corrupted_local = replace(corrupted_local, source_sha256=new_digest)
        invalid = evaluate_hard_guard(
            prediction,
            replace(clean_explanation, local_evidence=corrupted_local, canonical_sha256=new_digest),
            replace(clean_route, artifact_sha256=new_digest),
            replace(clean_context, expected_artifact_sha256=new_digest),
        )
        assert invalid.status is HardGuardStatus.BLOCKED
        assert "LOCAL_EFFECT:feature_effect_mismatch" in invalid.reason_codes

        structural_only = evaluate_hard_guard(
            prediction,
            replace(clean_explanation, local_evidence=corrupted_local, canonical_sha256=new_digest),
            replace(clean_route, artifact_sha256=new_digest, observed_local_effects=None),
            replace(
                clean_context,
                expected_artifact_sha256=new_digest,
                require_local_effect_verification=False,
            ),
        )
        assert structural_only.status is HardGuardStatus.CERTIFIED



def test_native_shap_incorrect_background_is_explicitly_blocked() -> None:
    from fuzzyxai.selective_observer.local_effects import hash_numeric_input

    dataset = load_breast_cancer()
    X = np.asarray(dataset.data, dtype=float)
    y = np.asarray(dataset.target, dtype=int)
    train, heldout = train_test_split(
        np.arange(len(y)), test_size=0.25, stratify=y, random_state=42
    )
    pipeline = make_pipeline(
        StandardScaler(), LogisticRegression(max_iter=1500, random_state=42)
    )
    pipeline.fit(X[train], y[train])
    scaler = pipeline.named_steps["standardscaler"]
    masker = shap.maskers.Independent(scaler.transform(X[train]))
    reference_raw = scaler.inverse_transform(
        np.asarray(masker.data, dtype=float).mean(axis=0).reshape(1, -1)
    )[0]
    feature_axis = tuple(str(s) for s in dataset.feature_names)
    positive = np.where(pipeline.predict(X[heldout]) == 1)[0][0]
    index = heldout[positive]
    prediction, explanation, route, context, _ = _model_evidence(
        "native_shap_background", pipeline, X[index], f"shap-bg-{index}",
        feature_axis, 42, reference_values=reference_raw,
    )
    original = explanation.local_evidence
    assert original is not None
    wrong_ref = hash_numeric_input(feature_axis, tuple(float(x) for x in scaler.mean_))
    assert wrong_ref != original.reference_sha256
    invalid_local = replace(original, reference_sha256=wrong_ref)
    digest = _canonical_local_digest(invalid_local)
    invalid_local = replace(invalid_local, source_sha256=digest)
    result = evaluate_hard_guard(
        prediction,
        replace(explanation, local_evidence=invalid_local, canonical_sha256=digest),
        replace(route, artifact_sha256=digest),
        replace(context, expected_artifact_sha256=digest),
    )
    assert result.status is HardGuardStatus.BLOCKED
    assert "LOCAL_REFERENCE:reference_sha256_mismatch" in result.reason_codes
