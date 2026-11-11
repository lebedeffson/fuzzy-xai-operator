"""Independent nonlinear Shapley game, provenance, and hard-guard regression.

These controlled tests check a deliberately specified empirical
interventional coalition game. They do not establish generic explanations,
causality or a production unknown-error detection rate.
"""

from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest
import shap
from fuzzyxai.practical_controller import (
    DeploymentContext,
    ExplanationArtifact,
    HardGuardStatus,
    PredictionArtifact,
    RouteArtifacts,
    evaluate_hard_guard,
)
from fuzzyxai.selective_observer import (
    LocalEvidenceStatus,
    LocalExplanationEvidence,
    collect_exact_interventional_shapley,
    hash_empirical_background,
    verify_exact_interventional_shapley,
)
from fuzzyxai.selective_observer.local_effects import (
    collect_linear_reference_effects,
    hash_numeric_input,
    verify_additive_local_effects,
)
from sklearn.datasets import load_diabetes
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split

from experiments.observer_local_evidence_benchmark import _canonical_local_digest

AXIS = ("x0", "x1", "x2")
SAMPLE = (1.5, 0.8, 2.0)
BACKGROUND = (
    (0.0, 0.0, 0.0),
    (1.0, -1.0, 0.0),
    (-1.0, 1.0, 1.0),
    (2.0, 2.0, 0.0),
)


def nonlinear_score(rows: np.ndarray) -> np.ndarray:
    return (
        rows[:, 0] * rows[:, 1]
        + 0.4 * rows[:, 0]
        - 0.2 * rows[:, 1]
        + 0.1 * rows[:, 2] ** 2
    )


def _scorer(rows: tuple[tuple[float, ...], ...]) -> tuple[float, ...]:
    return tuple(float(v) for v in nonlinear_score(np.asarray(rows, dtype=float)))


def _clean():
    measured = collect_exact_interventional_shapley(
        object_id="obj",
        model_version="nonlinear-model-1",
        target_class="regression_output",
        feature_names=AXIS,
        input_values=SAMPLE,
        background_rows=BACKGROUND,
        target_scores=_scorer,
    )
    local = LocalExplanationEvidence(
        object_id="obj",
        model_version="nonlinear-model-1",
        target_class="regression_output",
        scope="local",
        feature_names=AXIS,
        contributions=measured.attributions,
        source_sha256=None,
        input_sha256=measured.input_sha256,
        background_sha256=measured.background_sha256,
    )
    digest = _canonical_local_digest(local)
    local = replace(local, source_sha256=digest)
    pred = PredictionArtifact(
        object_id="obj",
        model_version="nonlinear-model-1",
        prediction="regression_output",
        confidence=0.9,
        probabilities=(1.0,),
        feature_names=AXIS,
        input_sha256=measured.input_sha256,
    )
    expl = ExplanationArtifact(
        canonical_sha256=digest,
        explainer_version="exact-shapley-v1",
        model_version="nonlinear-model-1",
        explain_plan_version="plan-v1",
        dictionary_version="features-v1",
        available_channels=("prediction", "attribution", "provenance"),
        local_evidence=local,
    )
    route = RouteArtifacts(
        preprocessing_version="prep-v1",
        calibration_version="cal-v1",
        reference_population="empirical-background-v1",
        schema_version="3-numeric-v1",
        artifact_sha256=digest,
        observed_provenance_channels=("prediction", "attribution", "provenance"),
        observed_exact_shapley=measured,
    )
    ctx = DeploymentContext(
        expected_model_version=pred.model_version,
        expected_preprocessing_version=route.preprocessing_version,
        expected_explainer_version=expl.explainer_version,
        expected_calibration_version="cal-v1",
        expected_reference_population=route.reference_population,
        expected_schema_version=route.schema_version,
        expected_explain_plan_version=expl.explain_plan_version,
        expected_dictionary_version=expl.dictionary_version,
        expected_artifact_sha256=digest,
        mandatory_provenance_channels=route.observed_provenance_channels,
        maximum_reduction_loss=0.1,
        policy_version="exact-shapley-v1",
        require_local_evidence=True,
        require_input_binding=True,
        require_exact_shapley_verification=True,
    )
    return pred, expl, route, ctx


def _with_local(pred, expl, route, ctx, local):
    digest = _canonical_local_digest(local)
    local = replace(local, source_sha256=digest)
    return (
        pred,
        replace(expl, local_evidence=local, canonical_sha256=digest),
        replace(route, artifact_sha256=digest),
        replace(ctx, expected_artifact_sha256=digest),
    )


def test_nonlinear_coalition_game_matches_actual_shap_exact_explainer() -> None:
    pred, expl, route, ctx = _clean()
    assert evaluate_hard_guard(pred, expl, route, ctx).status is HardGuardStatus.CERTIFIED
    evidence = route.observed_exact_shapley
    assert evidence is not None and evidence.attributions is not None
    masker = shap.maskers.Independent(np.asarray(BACKGROUND, dtype=float), max_samples=4)
    native = shap.ExactExplainer(nonlinear_score, masker)(
        np.asarray([SAMPLE], dtype=float)
    )
    native_phi = np.asarray(native.values[0], dtype=float)
    np.testing.assert_allclose(evidence.attributions, native_phi, rtol=0, atol=1e-9)
    assert abs(sum(evidence.attributions) - (evidence.explained_score - evidence.baseline_score)) < 1e-10
    assert evidence.score_evaluations == 2 ** len(AXIS) * len(BACKGROUND)


def test_rehashed_same_sum_corruption_is_invisible_to_structural_and_sum_baselines() -> None:
    pred, expl, route, ctx = _clean()
    assert expl.local_evidence is not None
    original = expl.local_evidence.contributions
    assert original is not None
    damaged = list(original)
    damaged[0] += 0.12
    damaged[1] -= 0.12
    corrupted = _with_local(
        pred, expl, route, ctx,
        replace(expl.local_evidence, contributions=tuple(damaged)),
    )
    checked = evaluate_hard_guard(*corrupted)
    assert checked.status is HardGuardStatus.BLOCKED
    assert "EXACT_SHAPLEY:shapley_attribution_mismatch" in checked.reason_codes
    measured = route.observed_exact_shapley
    assert measured is not None
    assert abs(sum(damaged) - (measured.explained_score - measured.baseline_score)) < 1e-8
    weaker = evaluate_hard_guard(
        corrupted[0], corrupted[1],
        replace(corrupted[2], observed_exact_shapley=None),
        replace(corrupted[3], require_exact_shapley_verification=False),
    )
    assert weaker.status is HardGuardStatus.CERTIFIED


@pytest.mark.parametrize(
    ("change", "status", "reason"),
    [
        ({"background_sha256": "f" * 64}, HardGuardStatus.BLOCKED, "background_sha256_mismatch"),
        ({"input_sha256": "f" * 64}, HardGuardStatus.BLOCKED, "input_sha256_mismatch"),
        ({"background_sha256": None}, HardGuardStatus.REVIEW_REQUIRED, "explanation_background_sha256"),
    ],
)
def test_background_and_input_bindings_are_not_silently_ignored(
    change, status, reason
) -> None:
    pred, expl, route, ctx = _clean()
    assert expl.local_evidence is not None
    new = _with_local(pred, expl, route, ctx, replace(expl.local_evidence, **change))
    outcome = evaluate_hard_guard(*new)
    assert outcome.status is status
    assert any(reason in code for code in (*outcome.reason_codes, *outcome.missing_evidence))


def test_required_missing_shapley_or_unsupported_game_requires_review() -> None:
    pred, expl, route, ctx = _clean()
    missing = evaluate_hard_guard(
        pred, expl, replace(route, observed_exact_shapley=None), ctx
    )
    assert missing.status is HardGuardStatus.REVIEW_REQUIRED
    unsupported = evaluate_hard_guard(
        pred, expl,
        replace(
            route,
            observed_exact_shapley=replace(
                route.observed_exact_shapley, protocol="conditional_shapley"
            ),
        ),
        ctx,
    )
    assert unsupported.status is HardGuardStatus.REVIEW_REQUIRED


@pytest.mark.parametrize(
    ("change", "expected_code"),
    [
        ({"target_class": "other"}, "target_class_mismatch"),
        ({"model_version": "stale"}, "model_version_mismatch"),
        ({"feature_names": ("x1", "x0", "x2")}, "feature_axis_mismatch"),
        ({"attributions": (0.1,)}, "attribution_width_mismatch"),
        ({"attributions": (float("nan"), 0.2, 0.3)}, "invalid_attribution_values"),
        ({"score_evaluations": 8}, "incompatible_probe_budget_metadata"),
    ],
)
def test_oracle_contract_invalid_evidence_never_certifies(change, expected_code) -> None:
    _pred, expl, route, _ctx = _clean()
    assert route.observed_exact_shapley is not None
    broken = replace(route.observed_exact_shapley, **change)
    observation = verify_exact_interventional_shapley(expl.local_evidence, broken)
    assert observation.status is LocalEvidenceStatus.INCOMPATIBLE
    assert expected_code in observation.contradictions


def test_interactions_require_nonadditive_protocol() -> None:
    # For f=x0*x1 with x=(1,1), B={(0,0)} the Shapley vector is
    # (0.5,0.5), whereas single-feature replacement yields (1,1).
    names = ("x0", "x1")
    def score(rows):
        return tuple(float(x0 * x1) for x0, x1 in rows)
    exact = collect_exact_interventional_shapley(
        object_id="interaction",
        model_version="f-v1",
        target_class="score",
        feature_names=names,
        input_values=(1.0, 1.0),
        background_rows=((0.0, 0.0),),
        target_scores=score,
    )
    assert exact.attributions == pytest.approx((0.5, 0.5))
    local = LocalExplanationEvidence(
        object_id="interaction",
        model_version="f-v1",
        target_class="score",
        scope="local",
        feature_names=names,
        contributions=exact.attributions,
        input_sha256=exact.input_sha256,
        background_sha256=exact.background_sha256,
    )
    assert verify_exact_interventional_shapley(
        local, exact
    ).status is LocalEvidenceStatus.CERTIFIED
    linear = collect_linear_reference_effects(
        object_id="interaction",
        model_version="f-v1",
        target_class="score",
        feature_names=names,
        input_values=(1.0, 1.0),
        reference_values=(0.0, 0.0),
        target_logit=lambda z: float(z[0] * z[1]),
    )
    assert verify_additive_local_effects(
        local, linear
    ).status is LocalEvidenceStatus.INCOMPATIBLE


def test_exact_background_hash_checks_whole_distribution_not_mean() -> None:
    first = ((-1.0, 0.0), (1.0, 0.0))
    second = ((0.0, -1.0), (0.0, 1.0))
    assert hash_empirical_background(("a", "b"), first) != hash_empirical_background(("a", "b"), second)
    assert hash_numeric_input(("a", "b"), (0.0, 0.0)) == hash_numeric_input(("a", "b"), (0.0, 0.0))


def test_equal_feature_means_do_not_make_nonlinear_backgrounds_equivalent() -> None:
    # Independent marginals have the same means, but opposite dependence:
    # positive and negative product backgrounds yield different game values.
    names = ("x0", "x1")
    positive = ((-1.0, -1.0), (1.0, 1.0))
    negative = ((-1.0, 1.0), (1.0, -1.0))
    assert np.allclose(np.mean(positive, axis=0), np.mean(negative, axis=0))
    assert hash_empirical_background(names, positive) != hash_empirical_background(
        names, negative
    )
    spec = {
        "object_id": "same-means",
        "model_version": "interaction-v1",
        "target_class": "output",
        "feature_names": names,
        "input_values": (1.5, 0.5),
        "target_scores": lambda rows: tuple(float(a * b) for a, b in rows),
    }
    first = collect_exact_interventional_shapley(
        **spec, background_rows=positive
    )
    second = collect_exact_interventional_shapley(
        **spec, background_rows=negative
    )
    assert first.baseline_score == pytest.approx(1.0)
    assert second.baseline_score == pytest.approx(-1.0)
    assert first.attributions == pytest.approx((-0.125, -0.125))
    assert second.attributions == pytest.approx((0.875, 0.875))
    assert first.explained_score == pytest.approx(second.explained_score)


def test_collector_rejects_excessive_cost_and_invalid_model_callback() -> None:
    with pytest.raises(ValueError, match="1..8 features"):
        collect_exact_interventional_shapley(
            object_id="x", model_version="v", target_class="y",
            feature_names=tuple(f"f{i}" for i in range(9)),
            input_values=(1.0,) * 9,
            background_rows=((0.0,) * 9,),
            target_scores=lambda _rows: (),
        )
    with pytest.raises(ValueError, match="budget"):
        collect_exact_interventional_shapley(
            object_id="x", model_version="v", target_class="y",
            feature_names=tuple(f"f{i}" for i in range(8)),
            input_values=(1.0,) * 8,
            background_rows=((0.0,) * 8,) * 33,
            target_scores=lambda _rows: (),
        )
    kwargs = {
        "object_id": "x",
        "model_version": "v",
        "target_class": "y",
        "feature_names": ("a", "b"),
        "input_values": (1.0, 2.0),
        "background_rows": ((0.0, 0.0), (1.0, 1.0)),
    }
    with pytest.raises(ValueError, match="score count"):
        collect_exact_interventional_shapley(
            **kwargs, target_scores=lambda _rows: (0.5,)
        )
    with pytest.raises(ValueError, match="finite scalar"):
        collect_exact_interventional_shapley(
            **kwargs,
            target_scores=lambda rows: (float("nan"),) * len(rows),
        )
    with pytest.raises(ValueError, match="non-deterministic"):
        collect_exact_interventional_shapley(
            **kwargs,
            target_scores=lambda rows: tuple(
                (10.0 + float(i)) if i >= len(rows) - 2 else float(sum(row))
                for i, row in enumerate(rows)
            ),
        )


def test_real_fitted_nonlinear_forest_matches_shap_exact_explainer() -> None:
    data = load_diabetes()
    x = np.asarray(data.data[:, :3], dtype=float)
    y = np.asarray(data.target, dtype=float)
    train, heldout = train_test_split(np.arange(len(y)), random_state=73, test_size=0.2)
    forest = RandomForestRegressor(n_estimators=12, max_depth=4, random_state=73)
    forest.fit(x[train], y[train])
    source = x[heldout[0]]
    bg = x[train[:6]]
    names = ("bmi_0", "bmi_1", "bmi_2")
    scores = lambda rows: tuple(
        float(v) for v in forest.predict(np.asarray(rows, dtype=float))
    )
    actual = collect_exact_interventional_shapley(
        object_id="diabetes-heldout-0",
        model_version="fitted-rf-v1",
        target_class="regression",
        feature_names=names,
        input_values=tuple(float(v) for v in source),
        background_rows=tuple(tuple(float(v) for v in row) for row in bg),
        target_scores=scores,
    )
    native = shap.ExactExplainer(
        forest.predict, shap.maskers.Independent(bg, max_samples=6)
    )(source.reshape(1, -1))
    np.testing.assert_allclose(actual.attributions, native.values[0], rtol=0, atol=1e-7)
    assert actual.score_evaluations == 48
