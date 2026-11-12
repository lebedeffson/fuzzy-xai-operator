"""Conditional linear-reference rebasing with genuine native SHAP attributions.

A numerical proposal is never an automatic certification: only after the
source-model effect probe is rerun at the new background can the full guard
accept the rebuilt explanation.
"""

from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest
import shap
from fuzzyxai.practical_controller import HardGuardStatus, evaluate_hard_guard
from fuzzyxai.selective_observer.local_effects import (
    ReferenceRebaseStatus,
    propose_linear_reference_rebase,
)
from sklearn.datasets import load_breast_cancer, load_digits, load_wine
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from experiments.observer_local_evidence_benchmark import _canonical_local_digest, _model_evidence


@pytest.mark.parametrize(
    ("name", "loader", "target_index"),
    [
        ("breast_cancer", load_breast_cancer, 0),
        ("breast_cancer", load_breast_cancer, 1),
        ("wine", load_wine, 0),
        ("wine", load_wine, 2),
        ("digits", load_digits, 0),
        ("digits", load_digits, 9),
    ],
)
def test_native_shap_rebase_is_a_proposal_until_new_model_probe_certifies(
    name: str, loader, target_index: int
) -> None:
    data = loader()
    features = np.asarray(data.data, dtype=float)
    labels = np.asarray(data.target)
    train, heldout = train_test_split(
        np.arange(len(labels)), test_size=0.25, stratify=labels, random_state=42
    )
    pipeline = make_pipeline(
        StandardScaler(), LogisticRegression(max_iter=1500, random_state=42)
    )
    pipeline.fit(features[train], labels[train])
    scaler = pipeline.named_steps["standardscaler"]
    model = pipeline.named_steps["logisticregression"]
    z_test = scaler.transform(features[heldout])
    index = np.flatnonzero(model.predict(z_test) == model.classes_[target_index])[0]
    original_row = heldout[index]
    maskers = {
        size: shap.maskers.Independent(scaler.transform(features[train]), max_samples=size)
        for size in (50, 100)
    }
    native = {
        size: np.asarray(shap.LinearExplainer(model, masker)(z_test[index : index + 1]).values, dtype=float)
        for size, masker in maskers.items()
    }
    refs = {
        size: scaler.inverse_transform(np.asarray(masker.data, dtype=float).mean(axis=0, keepdims=True))[0]
        for size, masker in maskers.items()
    }
    if len(model.classes_) == 2:
        sign = 1.0 if target_index == 1 else -1.0
        phi0 = native[50][0] * sign
        phi1 = native[100][0] * sign
        raw_coeff = sign * model.coef_[0] / scaler.scale_
    else:
        phi0 = native[50][0, :, target_index]
        phi1 = native[100][0, :, target_index]
        raw_coeff = model.coef_[target_index] / scaler.scale_

    axis = tuple(str(item) for item in data.feature_names)
    sample = features[original_row]
    pred0, expl0, route0, ctx0, _ = _model_evidence(
        name, pipeline, sample, f"{name}-rebase-{original_row}", axis, 42,
        reference_values=refs[50],
    )
    assert expl0.local_evidence is not None
    # Source attribution is from native SHAP, not the analytical helper.
    local0 = replace(
        expl0.local_evidence, contributions=tuple(float(v) for v in phi0)
    )
    proposal = propose_linear_reference_rebase(
        local0,
        feature_names=axis,
        input_values=tuple(float(v) for v in sample),
        original_reference=tuple(float(v) for v in refs[50]),
        proposed_reference=tuple(float(v) for v in refs[100]),
        signed_raw_coefficients=tuple(float(v) for v in raw_coeff),
    )
    assert proposal.status is ReferenceRebaseStatus.CANDIDATE, proposal
    assert proposal.contributions is not None
    assert proposal.original_max_residual is not None
    assert proposal.original_max_residual < 1e-8
    np.testing.assert_allclose(proposal.contributions, phi1, atol=1e-8, rtol=0)
    assert proposal.new_reference_sha256 != local0.reference_sha256

    # An incomplete proposal carrying a new background cannot pass using
    # *old* source-response evidence, even after a consistent rehash.
    intermediate = replace(
        local0,
        contributions=proposal.contributions,
        reference_sha256=proposal.new_reference_sha256,
    )
    intermediate_digest = _canonical_local_digest(intermediate)
    intermediate = replace(intermediate, source_sha256=intermediate_digest)
    wrong_route = replace(route0, artifact_sha256=intermediate_digest)
    wrong_exp = replace(
        expl0, local_evidence=intermediate, canonical_sha256=intermediate_digest
    )
    wrong_ctx = replace(ctx0, expected_artifact_sha256=intermediate_digest)
    blocked = evaluate_hard_guard(pred0, wrong_exp, wrong_route, wrong_ctx)
    assert blocked.status is HardGuardStatus.BLOCKED
    assert "LOCAL_REFERENCE:reference_sha256_mismatch" in blocked.reason_codes

    # Only independent source-model probing at the NEW reference completes
    # the transition; the caller must explicitly construct and rehash it.
    pred1, expl1, route1, ctx1, _ = _model_evidence(
        name, pipeline, sample, f"{name}-rebase-{original_row}", axis, 42,
        reference_values=refs[100],
    )
    renewed = replace(
        expl1.local_evidence, contributions=proposal.contributions
    )
    digest1 = _canonical_local_digest(renewed)
    renewed = replace(renewed, source_sha256=digest1)
    certified = evaluate_hard_guard(
        pred1,
        replace(expl1, local_evidence=renewed, canonical_sha256=digest1),
        replace(route1, artifact_sha256=digest1),
        replace(ctx1, expected_artifact_sha256=digest1),
    )
    assert certified.status is HardGuardStatus.CERTIFIED, certified.reason_codes


def test_rebase_rejects_missing_and_incompatible_original_evidence() -> None:
    from fuzzyxai.selective_observer.evidence_identity import LocalExplanationEvidence
    from fuzzyxai.selective_observer.local_effects import hash_numeric_input

    axis = ("a", "b")
    input_values = (2.0, 3.0)
    old = (0.0, 0.0)
    new = (1.0, 1.0)
    weights = (0.2, -0.1)
    local = LocalExplanationEvidence(
        object_id="one",
        model_version="linear-model",
        target_class="positive",
        scope="local",
        feature_names=axis,
        contributions=(0.4, -0.3),
        input_sha256=hash_numeric_input(axis, input_values),
        reference_sha256=hash_numeric_input(axis, old),
    )
    kwargs = {
        "feature_names": axis,
        "input_values": input_values,
        "original_reference": old,
        "proposed_reference": new,
        "signed_raw_coefficients": weights,
    }
    valid = propose_linear_reference_rebase(local, **kwargs)
    assert valid.status is ReferenceRebaseStatus.CANDIDATE
    assert valid.contributions == pytest.approx((0.2, -0.2))
    assert propose_linear_reference_rebase(
        replace(local, contributions=(0.5, -0.4)), **kwargs
    ).status is ReferenceRebaseStatus.INCOMPATIBLE
    missing = propose_linear_reference_rebase(
        replace(local, reference_sha256=None), **kwargs
    )
    assert missing.status is ReferenceRebaseStatus.INSUFFICIENT_EVIDENCE
    assert missing.contributions is None
    assert propose_linear_reference_rebase(
        replace(local, input_sha256="f" * 64), **kwargs
    ).status is ReferenceRebaseStatus.INCOMPATIBLE
    with pytest.raises(ValueError, match="tolerance"):
        propose_linear_reference_rebase(local, **kwargs, absolute_tolerance=-1)
