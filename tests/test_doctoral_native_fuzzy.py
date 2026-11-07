"""Measured fuzzy-rule inference + fail-closed proof checks on public WDBC."""
from __future__ import annotations

from copy import deepcopy

import numpy as np
import pytest
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split

from fuzzyxai.doctoral_certification import certify_payload, verify_certificate
from experiments.doctoral_native_fuzzy_logistic import NativeFuzzyLogistic
from experiments.doctoral_native_fuzzy_wdbc import make_receipt_payload, study

TAX = {"method_id": "native_fuzzy_wdbc_v1", "source": "intrinsic", "object": "logic",
       "audience": "auditor", "canonical_configuration": "measured native fuzzy rule predictor"}


@pytest.fixture(scope='module')
def model_data():
    data = load_breast_cancer()
    x = np.asarray(data.data, dtype=float)
    y = (data.target == 0).astype(int)
    train, test, yt, ys = train_test_split(x, y, test_size=.25, stratify=y, random_state=37)
    fitted = NativeFuzzyLogistic(5).fit(train, yt, list(data.feature_names))
    return fitted, train, test, yt, ys


def certificate(model, x, *, oid='wdbc_replay', origin='measured', allow_missing=False):
    payload, representation, metrics = make_receipt_payload(
        model, x, object_id=oid, dataset_version='wdbc:split37:measured',
        origin=origin, allow_missing=allow_missing,
        generated_at='2026-11-07T15:00:00+00:00')
    receipt = certify_payload(payload, taxonomy=TAX, representation=representation,
                              policy={"allow_controlled_fixture": origin == 'controlled_fixture'})
    return receipt, payload, representation, metrics


def issue_codes(receipt):
    return {v['code'] for v in receipt['decision']['issues']}


def test_predictor_uses_fuzzy_rule_activations_in_real_inference(model_data):
    model, _, test, _, _ = model_data
    x = test[0]
    out = model._explain(x)
    p = float(model.predict_proba(x)[0,1])
    assert out['probability'] == pytest.approx(p, abs=1e-12)
    assert out['gamma'] < 1e-12
    assert len(out['rules']) == 2 * len(model.features)
    assert out['delta'] == pytest.approx(abs(p - out['score_reduced']))
    for feature in model.features:
        parts = [r for r in out['rules'] if r['feature'] == feature.key]
        assert len(parts) == 2
        assert sum(v['membership'] for v in parts) == pytest.approx(1.0, abs=1e-12)
    # At least some actual fitted rule coefficients have learned nonzero effects.
    assert any(abs(v['contribution']) > 1e-4 for v in out['rules'])


def test_out_of_train_quantile_range_clips_before_membership_replay(model_data):
    model, _, test, _, _ = model_data
    outlier = test[0].copy()
    index = model.features[0].index
    outlier[index] = model.features[0].high * 2.0 + 100.0
    out = model._explain(outlier)
    terms = [v for v in out['rules'] if v['feature'] == model.features[0].key]
    assert sum(v['membership'] for v in terms) == pytest.approx(1.0)
    assert all(v['value'] == model.features[0].high for v in terms)
    assert all(v['raw_value'] == outlier[index] for v in terms)
    assert out['gamma'] < 1e-12
    receipt, *_ = certificate(model, outlier)
    assert 'D_native_gamma_replay' not in issue_codes(receipt)
    assert 'D_membership_replay' not in issue_codes(receipt)


def test_at_least_one_observed_case_gets_valid_native_model_receipt(model_data):
    model, _, test, _, _ = model_data
    outcomes = [certificate(model, test[i]) for i in range(min(20,len(test)))]
    assert any(v[0]['decision']['status']=='valid' for v in outcomes)
    for receipt, payload, representation, measured in outcomes:
        assert verify_certificate(receipt)['valid']
        native = receipt['decision']['native_model_replay']
        assert native['status']=='replayed'
        assert native['probability_gap'] < 1e-10
        assert measured['probability'] == pytest.approx(payload['model']['predicted_malignancy_probability'])


def test_prediction_tampering_is_detected_without_a_trusted_label(model_data):
    model, _, test, _, _ = model_data
    _, payload, rep, _ = certificate(model, test[0])
    forged = deepcopy(payload)
    forged['model']['predicted_malignancy_probability'] += 0.1
    output=certify_payload(forged, taxonomy=TAX, representation=rep)
    assert output['decision']['status']=='invalid'
    assert 'D_native_rule_reconstruction' in issue_codes(output)


def test_model_parameter_manifest_binding_detects_fake_coefficients(model_data):
    model, _, test, _, _ = model_data
    _, payload, rep, _ = certificate(model,test[0])
    forged=deepcopy(payload)
    forged['model']['parameter_manifest']['rules'][0]['low_consequent'] += 1.0
    output=certify_payload(forged,taxonomy=TAX,representation=rep)
    assert 'D_native_fingerprint' in issue_codes(output)
    assert 'D_native_rule_parameter' in issue_codes(output)


def test_rule_coefficient_change_without_new_model_detected(model_data):
    model, _, test, _, _ = model_data
    _, payload, rep, _=certificate(model,test[0])
    forged=deepcopy(payload)
    forged['layers']['rules'][0]['consequent'] *= -1.0
    output=certify_payload(forged,taxonomy=TAX,representation=rep)
    assert 'D_native_rule_parameter' in issue_codes(output)


def test_membership_projection_tamper_detected(model_data):
    model, _, test, _, _ = model_data
    _,payload,rep,_=certificate(model,test[0])
    key=next(iter(payload['fuzzy']['memberships']['wdbc_replay']))
    payload['fuzzy']['memberships']['wdbc_replay'][key]=0.222222
    result=certify_payload(payload,taxonomy=TAX,representation=rep)
    assert 'D_membership_projection_mismatch' in issue_codes(result)


def test_missing_feature_never_auto_accepts_or_hides_source_conflict(model_data):
    model, _, test, _, _=model_data
    missing=test[0].copy();missing[model.features[0].index]=np.nan
    with pytest.raises(ValueError):
        model.predict_proba(missing)
    receipt,payload,rep,metrics=certificate(model,missing,oid='controlled_dropout',origin='controlled_fixture',allow_missing=True)
    assert metrics['action']=='block'
    assert receipt['decision']['action']=='block'
    assert not receipt['decision']['auto_accept_permitted']
    assert payload['risk']['components']['chi_R']==1
    assert payload['risk']['chi_r_crit']==1
    assert rep['selected_class'] in {'F_N_src','not_certified'}
    assert verify_certificate(receipt)['valid']


def test_train_labels_affect_actual_coefficients(model_data):
    model, train, test, y_train, _=model_data
    inverted=NativeFuzzyLogistic(5).fit(train,1-y_train,model.feature_names)
    assert inverted.fingerprint()!=model.fingerprint()
    assert abs(inverted.predict_proba(test[0])[0,1]-model.predict_proba(test[0])[0,1])>.2


def test_single_seed_study_is_real_split_and_records_all_cases(tmp_path):
    output=study(tmp_path/'single',seed=42,n_features=5,model_factory=NativeFuzzyLogistic)
    assert output['n_train']==341 and output['n_validation']==114 and output['n_test']==114
    assert output['metric_comparison']['native_fuzzy']['n']==114
    assert sum(output['certification_status_counts'].values())==114
    assert output['stress_missing_feature']['all_actions_blocked']
    assert (tmp_path/'single'/'sample_certificate.json').exists()
    assert 0.75 < output['metric_comparison']['native_fuzzy']['roc_auc'] < 1.000001
