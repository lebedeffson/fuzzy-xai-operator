from __future__ import annotations

import sys
from dataclasses import replace
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from benchmarks.tisu_composition_validation import BETA, POLICY, SCENARIOS, TYPED_ONLY, build_pair, inject, term_specs, as_f0
from fuzzyxai.core.composition import compose
from fuzzyxai.core.composition_checked import (
    CompositionPolicy, MembershipSpec, compose_checked, result_status,
)


@pytest.mark.parametrize('scenario,expected', SCENARIOS)
@pytest.mark.parametrize('p', [0.13, 0.48, 0.714828385, 0.92])
def test_all_predeclared_scenarios(scenario, expected, p):
    a, b, origin = build_pair(12, p)
    a, b, origin = inject(scenario, a, b, origin)
    assert result_status(compose_checked(a, b, BETA, origin=origin, policy=POLICY)) == expected


def test_piecewise_distance_not_aliased_by_original_F0_grid():
    a, b, origin = build_pair(12, 0.70)
    a, b, origin = inject('grid_alias_membership', a, b, origin)
    assert a.representation.distance(b.representation) == pytest.approx(0.0, abs=1e-14)
    assert MembershipSpec.from_json(a.metadata['term_specs']['high']).distance(MembershipSpec.from_json(b.metadata['term_specs']['high'])) == pytest.approx(0.6)
    assert result_status(compose(a, b, BETA)) == 'ACCEPT'
    assert result_status(compose_checked(a, b, BETA, origin=origin)) == 'REJECT'


def test_breakpoint_union_exactness():
    a = MembershipSpec(((0.0, 0.0), (0.5, 0.0), (1.0, 1.0)))
    b = MembershipSpec(((0.0, 0.0), (0.005, 0.6), (0.01, 0.0),
                        (0.5, 0.0), (1.0, 1.0)))
    assert a.distance(b) == pytest.approx(0.6)
    assert b.distance(a) == pytest.approx(0.6)
    assert a.distance(a) == pytest.approx(0.0)
    with pytest.raises(ValueError):
        MembershipSpec(((0.0, 0.0), (0.0, 0.1), (1.0, 1.0)))
    with pytest.raises(ValueError):
        MembershipSpec(((0.0, 0.0), (0.5, 1.5), (1.0, 0.0)))


def test_valid_complement_result_is_canonical():
    a, b, origin = build_pair(12, 0.70)
    a, b, origin = inject('valid_complement', a, b, origin)
    result = compose_checked(a, b, BETA, origin=origin)
    assert result_status(result) == 'ACCEPT'
    assert result.metadata['target_event'] == 'malignant'
    assert result.metadata['probability'] == pytest.approx(0.70)
    assert result.metadata['sample_id'] == origin.sample_id
    assert set(result.metadata['term_specs']) == {'low', 'medium', 'high'}


def test_missing_origin_pairwise_tamper_undetectable():
    a, b, origin = build_pair(12, 0.70)
    a, b, origin = inject('coordinated_wrong_event', a, b, origin)
    no_origin_policy = replace(POLICY, require_origin_record=False)
    assert result_status(compose_checked(a, b, BETA, origin=origin)) == 'REJECT'
    assert result_status(compose_checked(a, b, BETA, origin=None, policy=no_origin_policy)) == 'ACCEPT'


def test_original_compose_api_stays_backward_compatible():
    a, b, origin = build_pair(12, 0.70)
    a, b, origin = inject('conflicting_rule_signature', a, b, origin)
    assert result_status(compose(a, b, BETA)) == 'ACCEPT'
    assert result_status(compose_checked(a, b, BETA, origin=origin)) == 'REJECT'


def test_incomplete_fuzzy_basis_is_not_silently_accepted():
    a, b, origin = build_pair(12, 0.70)
    del b.metadata['fuzzy_event']
    assert result_status(compose_checked(a, b, BETA, origin=origin)) == 'UNKNOWN'
    a, b, origin = build_pair(12, 0.70)
    del b.metadata['representation_term']
    assert result_status(compose_checked(a, b, BETA, origin=origin)) == 'UNKNOWN'


def test_mislabeled_rule_activation_is_rejected():
    a, b, origin = build_pair(12, 0.70)
    b.activations['risk_rule'] = 0.0
    assert result_status(compose_checked(a, b, BETA, origin=origin)) == 'REJECT'


def test_invalid_policy_is_rejected():
    for bad in (-0.01, 1.01, float('nan')):
        with pytest.raises(ValueError):
            CompositionPolicy(max_membership_distance=bad)


def test_real_predictions_across_two_model_families():
    from sklearn.datasets import load_breast_cancer
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import train_test_split

    dataset = load_breast_cancer()
    x_train, x_test, y_train, _ = train_test_split(
        dataset.data, (dataset.target == 0).astype(int),
        test_size=0.25, stratify=(dataset.target == 0).astype(int), random_state=42,
    )
    for name, model in (
        ('random_forest', RandomForestClassifier(n_estimators=40, max_depth=6, random_state=42)),
        ('logistic_regression', LogisticRegression(max_iter=300, solver='liblinear', random_state=42)),
    ):
        model.fit(x_train, y_train)
        positives = model.predict_proba(x_test)[:, list(model.classes_).index(1)]
        for ix in (0, 12, 42, 113):
            a, b, origin = build_pair(ix, float(positives[ix]), model_id=name)
            assert result_status(compose_checked(a, b, BETA, origin=origin)) == 'ACCEPT'
            a, b, origin = inject('valid_complement', a, b, origin)
            assert result_status(compose_checked(a, b, BETA, origin=origin)) == 'ACCEPT'
            a, b, origin = build_pair(ix, float(positives[ix]), model_id=name)
            a, b, origin = inject('grid_alias_membership', a, b, origin)
            assert result_status(compose_checked(a, b, BETA, origin=origin)) == 'REJECT'


def test_policy_thresholds_are_fixed_in_advance():
    assert POLICY.max_score_difference == 0.05
    assert POLICY.max_membership_distance == 0.08
    assert POLICY.require_origin_record
    assert POLICY.verify_membership
    assert not TYPED_ONLY.verify_membership


def test_non_certified_callable_is_unknown_even_if_it_matches_every_probe():
    from fuzzyxai.hierarchy.f0 import F0

    left, right, origin = build_pair(12, 0.70)
    baseline = right.representation.mu
    # Sub-grid perturbation cannot be excluded by a finite number of probes.
    right.representation = F0(
        lambda x: 0.55 if abs(x - 0.333333333) < 1e-9 else baseline(x),
        'high-risk',
    )
    assert result_status(compose_checked(left, right, BETA, origin=origin)) == 'UNKNOWN'


def test_malformed_membership_interface_fails_closed_without_exception():
    left, right, origin = build_pair(12, 0.70)
    right.metadata['representation_term'] = {'malformed': 'value'}
    assert result_status(compose_checked(left, right, BETA, origin=origin)) == 'REJECT'
    left, right, origin = build_pair(12, 0.70)
    right.metadata['term_specs']['high'] = [[0.0, 0.0], [1.0]]
    assert result_status(compose_checked(left, right, BETA, origin=origin)) == 'REJECT'
