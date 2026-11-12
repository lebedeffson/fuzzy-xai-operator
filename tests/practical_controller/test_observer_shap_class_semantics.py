"""Class-conditional native SHAP logit regression across binary and multiclass models.

Only independent-masker LinearExplainer and target *logit* attributions
are covered. This is a compatibility and rejection test, not general
SHAP faithfulness or causality validation.
"""

from __future__ import annotations

from dataclasses import replace
from functools import lru_cache

import numpy as np
import pytest
import shap
from fuzzyxai.practical_controller import HardGuardStatus, evaluate_hard_guard
from sklearn.datasets import load_breast_cancer, load_digits, load_wine
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from experiments.observer_local_evidence_benchmark import _canonical_local_digest, _model_evidence


@lru_cache(maxsize=3)
def _trained(dataset: str):
    loaders = {
        "breast_cancer": load_breast_cancer,
        "wine": load_wine,
        "digits": load_digits,
    }
    data = loaders[dataset]()
    features = np.asarray(data.data, dtype=float)
    labels = np.asarray(data.target)
    train, heldout = train_test_split(
        np.arange(len(labels)), test_size=0.25, stratify=labels, random_state=42
    )
    pipeline = make_pipeline(
        StandardScaler(), LogisticRegression(max_iter=1500, random_state=42)
    )
    pipeline.fit(features[train], labels[train])
    axis = tuple(str(item) for item in data.feature_names)
    return features, train, heldout, pipeline, axis


@pytest.mark.parametrize(
    ("dataset", "target_index"),
    [
        ("breast_cancer", 0),
        ("breast_cancer", 1),
        ("wine", 0),
        ("wine", 2),
        ("digits", 0),
        ("digits", 9),
    ],
)
@pytest.mark.parametrize("background_size", [50, 100])
def test_native_shap_selected_target_logit_matches_independent_model_probes(
    dataset: str, target_index: int, background_size: int
) -> None:
    features, train, heldout, pipeline, axis = _trained(dataset)
    scaler = pipeline.named_steps["standardscaler"]
    model = pipeline.named_steps["logisticregression"]
    classes = model.classes_
    masker = shap.maskers.Independent(
        scaler.transform(features[train]), max_samples=background_size
    )
    explainer = shap.LinearExplainer(model, masker)
    standardized = scaler.transform(features[heldout])
    predicted = model.predict(standardized)
    indices = np.flatnonzero(predicted == classes[target_index])
    assert len(indices), "target class absent from held-out predictions"
    source_index = heldout[indices[0]]
    source_values = standardized[indices[0] : indices[0] + 1]
    native_values = np.asarray(explainer(source_values).values, dtype=float)
    if len(classes) == 2:
        assert native_values.shape == (1, len(axis))
        signed = native_values[0] * (1.0 if target_index == 1 else -1.0)
        incorrect = -signed
    else:
        assert native_values.shape == (1, len(axis), len(classes))
        signed = native_values[0, :, target_index]
        incorrect = native_values[0, :, (target_index + 1) % len(classes)]

    reference_z = np.asarray(masker.data, dtype=float).mean(axis=0)
    reference_raw = scaler.inverse_transform(reference_z.reshape(1, -1))[0]
    prediction, explanation, route, context, _ = _model_evidence(
        f"shap_{dataset}",
        pipeline,
        features[source_index],
        f"{dataset}-{target_index}-{background_size}-{source_index}",
        axis,
        42,
        reference_values=reference_raw,
    )
    assert prediction.prediction == str(classes[target_index])
    effects = route.observed_local_effects
    assert effects is not None and effects.effect_on_score is not None
    np.testing.assert_allclose(signed, effects.effect_on_score, atol=1e-8, rtol=0)
    assert np.max(np.abs(signed - incorrect)) > 1e-5

    old_local = explanation.local_evidence
    assert old_local is not None
    clean_local = replace(old_local, contributions=tuple(float(x) for x in signed))
    correct_digest = _canonical_local_digest(clean_local)
    clean_local = replace(clean_local, source_sha256=correct_digest)
    clean_explanation = replace(
        explanation, local_evidence=clean_local, canonical_sha256=correct_digest
    )
    clean_route = replace(route, artifact_sha256=correct_digest)
    clean_context = replace(context, expected_artifact_sha256=correct_digest)
    accepted = evaluate_hard_guard(
        prediction, clean_explanation, clean_route, clean_context
    )
    assert accepted.status is HardGuardStatus.CERTIFIED, accepted.reason_codes

    # Independently rehash a valid-shape but *wrong target-class* explanation.
    wrong_local = replace(clean_local, contributions=tuple(float(x) for x in incorrect))
    wrong_digest = _canonical_local_digest(wrong_local)
    wrong_local = replace(wrong_local, source_sha256=wrong_digest)
    wrong_explanation = replace(
        clean_explanation, local_evidence=wrong_local, canonical_sha256=wrong_digest
    )
    wrong_route = replace(clean_route, artifact_sha256=wrong_digest)
    wrong_context = replace(clean_context, expected_artifact_sha256=wrong_digest)
    rejected = evaluate_hard_guard(
        prediction, wrong_explanation, wrong_route, wrong_context
    )
    assert rejected.status is HardGuardStatus.BLOCKED
    assert "LOCAL_EFFECT:feature_effect_mismatch" in rejected.reason_codes

    structural = evaluate_hard_guard(
        prediction,
        wrong_explanation,
        replace(wrong_route, observed_local_effects=None),
        replace(
            wrong_context,
            require_local_effect_verification=False,
            require_reference_binding=False,
        ),
    )
    assert structural.status is HardGuardStatus.CERTIFIED, structural.reason_codes
