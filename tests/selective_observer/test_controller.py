from __future__ import annotations

import numpy as np

from fuzzyxai.selective_observer import ConfirmatoryProtocolLock

from fuzzyxai.selective_observer import (
    ConfirmatoryExample,
    DevelopmentExample,
    ResearchPartition,
    SelectiveAction,
    SelectiveRiskFeatures,
    compare_preregistered_baselines,
    confidence_threshold_policy,
    decide,
    evaluate_confirmatory_policy,
    fit_selective_controller,
    risk_coverage_curve,
)


def _features(risk: float, *, rupture: float = 0.0) -> SelectiveRiskFeatures:
    return SelectiveRiskFeatures(
        model_uncertainty=risk,
        calibration_residual=0.8 * risk,
        boundary_proximity=0.9 * risk,
        model_disagreement=0.7 * risk,
        explainer_disagreement=0.6 * risk,
        attribution_instability=0.5 * risk,
        provenance_incompleteness=0.4 * risk,
        data_shift=0.3 * risk,
        representation_loss=0.6 * risk,
        rupture_severity=rupture,
        rare_group=float(risk > 0.7),
    )


def test_controller_uses_oof_development_and_keeps_confirmation_separate(
    protocol_lock: ConfirmatoryProtocolLock,
) -> None:
    examples = [
        DevelopmentExample(
            object_id=str(index),
            features=_features(index / 79),
            unsafe_automatic_action=index >= 48,
            partition=ResearchPartition.TRAIN if index % 2 else ResearchPartition.VALIDATION,
            source_features_are_oof=True,
            group_id=f"fold-{index % 5}",
        )
        for index in range(80)
    ]
    spec, formative = fit_selective_controller(examples)
    assert spec.selected_without_test is True
    assert formative["confirmatory_claim_allowed"] is False
    assert len(spec.development_hash) == 64
    assert decide(spec, _features(0.99, rupture=1.0)) is SelectiveAction.BLOCK

    test = [ConfirmatoryExample(str(index), _features(index / 39), unsafe_automatic_action=index >= 24) for index in range(40)]
    result = evaluate_confirmatory_policy(spec, test, protocol_lock)
    assert result["phase"] == "confirmatory"
    assert result["thresholds_frozen_before_test"] is True


def test_controller_rejects_non_oof_development_features() -> None:
    try:
        DevelopmentExample(
            object_id="leaky",
            features=_features(0.5),
            unsafe_automatic_action=False,
            partition=ResearchPartition.TRAIN,
            source_features_are_oof=False,
            group_id="fold-0",
        )
    except ValueError as error:
        assert "out-of-fold" in str(error)
    else:
        raise AssertionError("non-OOF development evidence was accepted")


def test_risk_coverage_curve_is_monotonic_in_coverage() -> None:
    curve = risk_coverage_curve(np.linspace(0.0, 1.0, 20), [index >= 12 for index in range(20)])
    coverage = [float(row["coverage"]) for row in curve]
    assert coverage == sorted(coverage)
    assert coverage[-1] == 1.0


def test_preregistered_baseline_comparison_does_not_create_formative_claim() -> None:
    features = [_features(index / 39) for index in range(40)]
    outcomes = [index >= 28 for index in range(40)]
    controller = [SelectiveAction.ACCEPT if index < 28 else SelectiveAction.FULL_REVIEW for index in range(40)]
    baseline = confidence_threshold_policy(features, confidence_threshold=0.70)
    result = compare_preregistered_baselines(outcomes, controller, {"confidence": baseline})
    assert result["claim_allowed"] is False
    assert str(result["reason"]).startswith("operating points are formative")
