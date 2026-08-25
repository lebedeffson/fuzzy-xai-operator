"""Locks in the invariant: contribution sign is relative to the predicted class.

Found via manual review of a demo explanation: for a binary classifier that
predicted classes_[0], sklearn's single stored coefficient row is relative to
classes_[1] by convention, and every downstream consumer of `contributions`
(claim building, HumanExplanation text, text highlighting, tabular view,
why_not()) assumes `contribution >= 0` means "pushed the model toward the
predicted class". Before the fix, a positive-coefficient feature was labeled
"supports" even when the model predicted the *other* class — backwards.

This file specifically covers the case earlier tests never covered: a binary
classifier whose predicted class is classes_[0], not classes_[1].
"""

from __future__ import annotations

import numpy as np
from fuzzyxai import FuzzyXAI
from sklearn.linear_model import LogisticRegression


def _fit_two_feature_binary_model() -> LogisticRegression:
    # feature_0 large -> class 1; feature_1 large -> class 0. Fully
    # controlled so the expected sign of each contribution is known exactly,
    # not just "whatever sklearn happens to compute".
    X = np.array([[0.0, 5.0], [0.0, 6.0], [0.0, 7.0], [5.0, 0.0], [6.0, 0.0], [7.0, 0.0]])
    y = np.array([0, 0, 0, 1, 1, 1])
    return LogisticRegression().fit(X, y)


def test_positive_class_prediction_keeps_conventional_sign() -> None:
    model = _fit_two_feature_binary_model()
    assert list(model.classes_) == [0, 1]
    fx = FuzzyXAI.wrap(model, adapter="auto", task="classification")

    result = fx.explain_one(np.array([6.5, 0.0]), object_id="predicts-class-1")
    assert result.prediction.predictions == [1]
    contributions = result.view_model.model["contributions"]
    # feature_0 drove the class-1 prediction; must be labeled as supporting it.
    assert contributions["feature_0"] > 0
    assert contributions["feature_1"] <= 0


def test_negative_class_prediction_flips_sign_relative_to_predicted_class() -> None:
    """The case the bug was found in: predicted class is classes_[0]."""

    model = _fit_two_feature_binary_model()
    fx = FuzzyXAI.wrap(model, adapter="auto", task="classification")

    result = fx.explain_one(np.array([0.0, 6.5]), object_id="predicts-class-0")
    assert result.prediction.predictions == [0]
    contributions = result.view_model.model["contributions"]
    # feature_1 drove the class-0 prediction; must be POSITIVE (supports the
    # predicted class), even though sklearn's raw coef_ for feature_1 is
    # negative relative to classes_[1]. This is exactly the case that was
    # inverted before the fix.
    assert contributions["feature_1"] > 0
    assert contributions["feature_0"] <= 0


def test_claim_effect_direction_matches_predicted_class_not_fixed_reference_class() -> None:
    """End-to-end: the claim-level effect (favorable/adverse) must follow
    the same prediction-relative convention, not just the raw contribution."""

    model = _fit_two_feature_binary_model()
    fx = FuzzyXAI.wrap(model, adapter="auto", task="classification")

    result = fx.explain_one(np.array([0.0, 6.5]), object_id="predicts-class-0")
    assert result.prediction.predictions == [0]
    feature_1_claims = [claim for claim in result.claims if claim.claim_type == "feature_contribution" and "feature_1" in claim.subject_id]
    assert feature_1_claims, "expected a feature_contribution claim for feature_1"
    # feature_1 is what actually drove the (correctly predicted) class-0
    # outcome, so its claim must be "favorable" (supports the prediction),
    # not "adverse".
    assert feature_1_claims[0].effect == "favorable"


def test_human_explanation_reason_direction_follows_predicted_class() -> None:
    model = _fit_two_feature_binary_model()
    fx = FuzzyXAI.wrap(model, adapter="auto", task="classification")

    result = fx.explain_one(np.array([0.0, 6.5]), object_id="predicts-class-0")
    human = result.explain_for(audience="ml_engineer")
    # HumanExplanation humanizes identifiers for display (underscore -> space).
    feature_1_reasons = [reason for reason in human.main_reasons if "feature 1" in reason.subject_label]
    assert feature_1_reasons
    assert feature_1_reasons[0].effect_direction == "supports"


def test_symmetric_binary_dataset_produces_opposite_signs_for_the_same_feature() -> None:
    """Direct before/after contrast on the same feature across both predicted classes."""

    model = _fit_two_feature_binary_model()
    fx = FuzzyXAI.wrap(model, adapter="auto", task="classification")

    result_class_1 = fx.explain_one(np.array([6.5, 0.0]), object_id="a")
    result_class_0 = fx.explain_one(np.array([0.0, 6.5]), object_id="b")
    assert result_class_1.prediction.predictions == [1]
    assert result_class_0.prediction.predictions == [0]

    # feature_1's raw coefficient sign never changes — only which class was
    # predicted changes — so its contribution sign must flip between the two
    # explanations, tracking the predicted class each time.
    c1 = result_class_1.view_model.model["contributions"]["feature_1"]
    c0 = result_class_0.view_model.model["contributions"]["feature_1"]
    assert (c1 <= 0) and (c0 > 0)
