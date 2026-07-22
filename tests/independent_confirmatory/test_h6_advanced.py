from __future__ import annotations

import numpy as np

from fuzzyxai.rule_effects_v2 import (
    audit_conditional_sampler,
    binary_effect_power,
    cluster_equivalent_rules,
    cross_fitted_doubly_robust_effect,
    evaluate_h6_formative_gate,
)


def test_rule_equivalence_clusters_redundant_activations() -> None:
    activations = {
        "r1": [1, 1, 0, 0, 1, 0],
        "r2": [1, 1, 0, 0, 1, 0],
        "r3": [0, 0, 1, 1, 0, 1],
    }
    clusters = cluster_equivalent_rules(activations)
    assert any(cluster.rule_ids == ("r1", "r2") for cluster in clusters)


def test_power_gate_fails_when_detection_is_weak() -> None:
    power = binary_effect_power(effect=0.1, support=0.2, total_objects=20_000)
    assert power.required_objects > 0
    gate = evaluate_h6_formative_gate(detection_rate=0.79, sign_accuracy=0.95, false_discovery_rate=0.05, power_eligible_fraction=0.8)
    assert not gate.confirmatory_opening_allowed


def test_cross_fitted_dr_recovers_positive_effect_without_in_sample_predictions() -> None:
    rng = np.random.default_rng(11)
    values = rng.normal(size=(600, 4))
    propensity = 1.0 / (1.0 + np.exp(-values[:, 0]))
    treatment = rng.binomial(1, propensity)
    outcome = 0.4 * treatment + values[:, 1] + rng.normal(0.0, 0.2, size=len(values))
    result = cross_fitted_doubly_robust_effect(values, treatment, outcome, folds=3)
    assert result.source_predictions_oof
    assert result.confidence_interval_95[0] > 0.0


def test_conditional_sampler_audit_detects_large_shift() -> None:
    rng = np.random.default_rng(13)
    reference = rng.normal(size=(200, 3))
    good = reference + rng.normal(0.0, 0.01, size=reference.shape)
    bad = reference + 5.0
    assert audit_conditional_sampler(reference, good).passed
    assert not audit_conditional_sampler(reference, bad).passed


def test_chronological_replay_has_bursts_and_low_irreparable_incidence() -> None:
    from fuzzyxai.replay import registered_incident_schedule, stream_chronological_events

    schedule = registered_incident_schedule(20_000)
    events = tuple(stream_chronological_events(20_000, incidents=schedule))
    incident_rate = np.mean([bool(item.active_incidents) for item in events])
    irreparable_rate = np.mean([bool(item.active_incidents) and not item.repairable for item in events])
    assert 0.0 < incident_rate < 0.20
    assert irreparable_rate <= 0.05
    assert len({item.model_lane for item in events}) == 3
