"""Acceptance tests requiring the complete FuzzyXAI source checkout.

These intentionally cannot be run from the isolated overlay: the claim that the
real model and the certification API interoperate requires the actual runtime.
"""
from __future__ import annotations

import math

import numpy as np
import pytest
from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression

from fuzzyxai import FuzzyXAI, verify_certificate
from fuzzyxai.core.alignment import compute_gamma_route
from fuzzyxai.core.risk_observer import compute_canonical_risk, compute_action
from fuzzyxai.risk.risk_function import compute_application_risk, expected_action_costs, normalize_risk_weights


def test_certify_actual_model_output_and_preserve_original_inference():
    x, y = make_classification(n_samples=90, n_features=6, n_informative=4, random_state=73)
    model = LogisticRegression(max_iter=500).fit(x[:65], y[:65])
    fx = FuzzyXAI.wrap(model)
    explain_kwargs = {'object_id': 'real-model-0', 'feature_names': [f'x{i}' for i in range(6)]}
    expected = fx.explain_one(x[65].tolist(), **explain_kwargs)
    actual, cert = fx.explain_one_certified(
        x[65].tolist(), taxonomy={'method_id':'m01',
            'source':'intrinsic', 'object':'features', 'audience':'auditor',
            'canonical_configuration':'Линейные и логистические модели'},
        representation={'profile':['u_num','u_ling'],'selected_class':'F0'},
        **explain_kwargs,
    )
    assert actual.prediction.predictions == expected.prediction.predictions
    assert actual.adapter_id == expected.adapter_id
    assert verify_certificate(cert)['valid'] is True
    assert cert['decision']['status'] in {'invalid', 'insufficient_evidence', 'valid'}
    # Ordinary logistic regression has not produced native fuzzy rules or an
    # executed F0 route by itself; the gate MUST NOT invent those channels.
    assert cert['decision']['status'] != 'valid'
    assert cert['decision']['auto_accept_permitted'] is False
    assert 'source model correctness' in cert['decision']['not_validated']


def test_export_and_replay_existing_audit_without_raw_payloads(tmp_path):
    x, y = make_classification(n_samples=60, n_features=5, random_state=6)
    model = LogisticRegression(max_iter=300).fit(x, y)
    result = FuzzyXAI.wrap(model).explain_one(x[10].tolist(), object_id='patient-pseudonym-10')
    payload = result.view_model.to_dict(include_raw=False)
    assert payload['trace']['object_ids'] == ['patient-pseudonym-10']
    receipt = result.certify(
        taxonomy={'method_id': 'm01', 'source':'intrinsic', 'object':'features',
                  'audience':'auditor','canonical_configuration':'Линейные и логистические модели'},
        representation={'profile':['u_num','u_ling'],'selected_class':'F0'},
    )
    assert verify_certificate(receipt)['valid']
    assert receipt['decision']['trace']['input_sha256'] == payload['trace']['input_sha256']


def test_hardened_risk_and_alignment_reject_numerical_corruption():
    weights = {'rho_p':.3,'u_M':.2,'one_minus_I_pre':.2,'Delta':.2,'chi_R':.1}
    components={'rho_p':.3,'u_M':.3,'one_minus_I_pre':.4,'Delta':.2,'chi_R':0.0}
    assert math.isfinite(compute_canonical_risk(components, weights))
    with pytest.raises(ValueError):
        compute_canonical_risk({**components, 'u_M': float('nan')}, weights)
    with pytest.raises(ValueError):
        compute_canonical_risk(components, {**weights,'u_M':float('inf')})
    with pytest.raises(ValueError):
        compute_action(0.1, 2, {'theta_1': .2,'theta_2': .4,'theta_3': .6,'theta_4': .8})
    assert compute_gamma_route(.67,.72,.47) == pytest.approx(.2)
    assert compute_gamma_route(.67,.31,.60) == pytest.approx(.36)
    with pytest.raises(ValueError):
        compute_gamma_route(.67,float('nan'),.47)


def test_application_risk_and_expected_costs_reject_numerical_corruption():
    with pytest.raises(ValueError):
        compute_application_risk(0.4, float('nan'), .6, 0.05)
    with pytest.raises(ValueError):
        normalize_risk_weights({'predicted_risk': float('nan')})
    with pytest.raises(ValueError):
        normalize_risk_weights({'predicted_risk': -0.2})
    with pytest.raises(ValueError):
        normalize_risk_weights({'unsupported':0.0})
    with pytest.raises(ValueError):
        expected_action_costs([0.5,float('nan')], {'accept':[0,1]})
    with pytest.raises(ValueError):
        expected_action_costs([.5,.5], {'accept':[0,float('inf')]})
