from __future__ import annotations

import copy
import math

import pytest

from fuzzyxai.interval_type2 import (
    IT2FeatureRule, IT2FuzzyLogit, IT2Shoulder, MembershipInterval,
    content_sha256, karnik_mendel_center_of_sets, make_it2_receipt,
    robust_interval_decision, saturated_high, verify_it2_receipt,
)


def demo() -> IT2FuzzyLogit:
    return IT2FuzzyLogit(
        intercept=-0.3, n_input_features=2,
        features=(IT2FeatureRule(0, "x0", -0.6, 1.0,
            IT2Shoulder(0.0, 0.2, 0.8, 1.1, 0.1, 0.95)),
            IT2FeatureRule(1, "x1", 0.8, -0.4,
            IT2Shoulder(0.0, 0.1, 0.7, 1.0, 0.05, 0.85))),
    )


def test_intervals_and_shoulders():
    s = demo().features[0].membership
    for x in [-2, 0, 0.15, 0.5, 0.85, 10]:
        high = s.high(x)
        low = s.low(x)
        assert 0 <= high.lower <= high.upper <= 1
        assert low.lower == pytest.approx(1 - high.upper)
        assert low.upper == pytest.approx(1 - high.lower)
        assert high.lower <= s.nominal_high(x) <= high.upper
    with pytest.raises(ValueError):
        MembershipInterval(-0.1, 0.3)
    with pytest.raises(ValueError):
        IT2Shoulder(0, 0.8, 0.6, 1.1, .1, .9)


def test_exact_additive_bounds_on_corner_mesh():
    m = demo()
    for row in ((0.2, 0.7), (0.5, 0.2), (0.8, 0.95), (-4, 12)):
        got = m.infer(row)
        samples = []
        for a0 in (m.features[0].membership.left_min, m.features[0].membership.left_max):
            for b0 in (m.features[0].membership.right_min, m.features[0].membership.right_max):
                for a1 in (m.features[1].membership.left_min, m.features[1].membership.left_max):
                    for b1 in (m.features[1].membership.right_min, m.features[1].membership.right_max):
                        z = m.intercept
                        for j, (a, b) in enumerate(((a0, b0), (a1, b1))):
                            mu = saturated_high(row[j], a, b)
                            f = m.features[j]
                            z += f.low_coefficient * (1-mu) + f.high_coefficient * mu
                        samples.append(z)
        assert got['logit_lower'] == pytest.approx(min(samples))
        assert got['logit_upper'] == pytest.approx(max(samples))
        assert got['probability_lower'] <= got['probability_nominal_type1'] <= got['probability_upper']


def test_receipt_replay_rejects_numeric_and_identity_tampering():
    m = demo()
    rec = make_it2_receipt(m, [.42, .7], source_id='record:42', dataset_version='toy-v1')
    assert verify_it2_receipt(rec, trusted_model_sha256=content_sha256(m.to_dict()))['valid']
    edits = (
        ('inference', 'probability_upper', .999),
        ('decision', 'action', 'block'),
        ('model_manifest', 'intercept', 19.0),
    )
    for top, key, new_value in edits:
        bad = copy.deepcopy(rec)
        bad[top][key] = new_value
        assert not verify_it2_receipt(bad, trusted_model_sha256=content_sha256(m.to_dict()))['valid']
    for key, value in [('input_values', [0, 0]), ('source_id', 'fake')]:
        bad = copy.deepcopy(rec)
        bad[key] = value
        assert not verify_it2_receipt(bad, trusted_model_sha256=content_sha256(m.to_dict()))['valid']


def test_abstain_and_safety():
    model = demo()
    crossing = model.infer([0.5, .5])
    assert robust_interval_decision(crossing, missing_source=True)['action'] == 'block'
    assert robust_interval_decision({'probability_lower': .3, 'probability_upper': .7,
           'probability_nominal_type1': .4})['action'] == 'request_more_data'
    assert robust_interval_decision({'probability_lower': .7, 'probability_upper': .95,
           'probability_nominal_type1': .8}, max_width=.1)['action'] == 'defer_to_human'
    assert robust_interval_decision({'probability_lower': .7, 'probability_upper': .75,
           'probability_nominal_type1': .73}, max_width=.1)['action'] == 'accept'


def test_km_reference_degenerate_crisp_case():
    l, r = karnik_mendel_center_of_sets([0., 1.], [1., 1.], [1., 1.])
    assert l == pytest.approx(.5)
    assert r == pytest.approx(.5)
    with pytest.raises(ValueError):
        karnik_mendel_center_of_sets([1.], [0.], [0.])


def test_manifest_rejects_nan_inf_and_duplicate_features():
    m = demo()
    for nan in (float('nan'), float('inf')):
        with pytest.raises(ValueError):
            IT2FuzzyLogit(nan, 2, m.features)
        with pytest.raises(ValueError):
            m.infer([nan, 0.5])
    with pytest.raises(ValueError):
        IT2FuzzyLogit(0, 2, (m.features[0], m.features[0]))

def test_it2_public_package_exports():
    from fuzzyxai import (
        IT2FuzzyLogit, IT2FeatureRule, IT2Shoulder, MembershipInterval,
        make_it2_receipt, verify_it2_receipt, robust_interval_decision,
        karnik_mendel_center_of_sets,
    )
    assert IT2FuzzyLogit.__name__ == "IT2FuzzyLogit"
    assert IT2FeatureRule.__name__ == "IT2FeatureRule"
    assert IT2Shoulder.__name__ == "IT2Shoulder"
    assert MembershipInterval.__name__ == "MembershipInterval"
    assert all(callable(item) for item in (
        make_it2_receipt, verify_it2_receipt, robust_interval_decision,
        karnik_mendel_center_of_sets,
    ))
