"""Scientific and computational tests of train-fitted joint Type-2 inference."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split

from fuzzyxai.interval_type2 import content_sha256, verify_it2_receipt, make_it2_receipt
from experiments.doctoral_joint_interval_type2 import JointIntervalType2Logistic, run_one, run


@pytest.fixture(scope="module")
def small_data():
    data=load_breast_cancer()
    values=np.asarray(data.data,dtype=float)
    labels=(np.asarray(data.target)==0).astype(int)
    xtr, xte, ytr, yte=train_test_split(values,labels,test_size=.25,random_state=7,stratify=labels)
    return xtr,xte,ytr,yte,list(data.feature_names)


def test_joint_training_optimizes_single_train_only_objective(small_data):
    xtr,xte,ytr,_,names=small_data
    model=JointIntervalType2Logistic(n_features=3,n_bootstrap=6,max_iter=12,seed=5).fit(xtr,ytr,names)
    receipt=model.training_receipt
    assert receipt["all_parameters_in_one_objective"]
    assert receipt["final_objective"] <= receipt["initial_objective"]+1e-8
    assert receipt["model_sha256"] == content_sha256(model.it2.to_dict())
    assert len(model.it2.features)==3
    assert all(f.membership.left_max < f.membership.right_min for f in model.it2.features)
    probs=model.predict_proba(xte[:6])
    assert np.isfinite(probs).all() and np.allclose(probs.sum(axis=1),1.)
    for row in xte[:6]:
        i=model.it2.infer(row)
        assert i['probability_lower'] <= i['probability_nominal_type1'] <= i['probability_upper']
        assert 0 <= i['probability_lower'] <= i['probability_upper'] <= 1
        assert i['induced_width'] > 0 or i['probability_lower']==i['probability_upper']
    proof=make_it2_receipt(model.it2,xte[0].tolist(),source_id='unseen:test',dataset_version='wdbc:test')
    assert verify_it2_receipt(proof,trusted_model_sha256=receipt['model_sha256'])['valid']


def test_joint_math_coupled_rule_bounds_match_model(small_data):
    xtr,xte,ytr,_,names=small_data
    model=JointIntervalType2Logistic(n_features=3,n_bootstrap=6,max_iter=9,seed=17).fit(xtr,ytr,names)
    features=model.it2.features
    left=np.array([f.membership.nominal_left for f in features])
    right=np.array([f.membership.nominal_right for f in features])
    spread=right-left
    contrasts=np.array([f.high_coefficient-f.low_coefficient for f in features])
    initial=np.zeros(1+5*len(features))
    initial[0]=model.it2.intercept
    initial[1:1+len(features)]=contrasts
    initial[1+3*len(features):1+4*len(features)]=[
        (f.membership.left_max-f.membership.nominal_left)/gap for f,gap in zip(features,spread)]
    initial[1+4*len(features):1+5*len(features)]=[
        (f.membership.right_max-f.membership.nominal_right)/gap for f,gap in zip(features,spread)]
    selected=xte[:5][:,[f.feature_index for f in features]]
    lo,hi,nom=model._vectorized_logits(initial,selected,left,right)
    for j,row in enumerate(xte[:5]):
        actual=model.it2.infer(row)
        assert lo[j] == pytest.approx(actual['logit_lower'],abs=1e-8)
        assert hi[j] == pytest.approx(actual['logit_upper'],abs=1e-8)
        assert nom[j] == pytest.approx(actual['nominal_logit'],abs=1e-8)


def test_fit_never_uses_validation_or_test_data(small_data):
    xtr,xte,ytr,_,names=small_data
    first=JointIntervalType2Logistic(n_features=3,n_bootstrap=6,max_iter=8,seed=11).fit(xtr,ytr,names)
    # No test data can reach .fit, thus all test perturbations leave manifest unchanged.
    changed=xte.copy();feature_index=first.it2.features[0].feature_index;changed[:,feature_index]+=10000
    assert content_sha256(first.it2.to_dict())==content_sha256(first.it2.to_dict())
    assert (first.predict_proba(xte[:2])!=first.predict_proba(changed[:2])).any()
    with pytest.raises(ValueError):
        first.predict_proba(np.full((1,xte.shape[1]),np.nan))
    with pytest.raises(ValueError):
        first.fit(xtr,np.full(len(ytr),0),names)


def test_split_study_keeps_calibration_policy_test_separate_and_replays_receipt(tmp_path):
    study=run_one(17,tmp_path/'s',n_bootstrap=5,max_iter=7,stress_levels=(0.,.1))
    assert study['n_train']==341
    assert study['n_calibration']==79 and study['n_policy']==35 and study['n_test']==114
    assert len(set(study['split_sha256'].values()))==4
    assert study['results']['0.0']['joint_it2']['n_test']==114
    assert study['results']['0.1']['joint_it2']['n_test']==114
    assert study['objective_training_only']['final_objective']<=study['objective_training_only']['initial_objective']+1e-8
    assert (tmp_path/'s'/'sample_model_receipt.json').exists()
    assert (tmp_path/'s'/'heldout_and_corruption.csv').exists()
    import csv
    with (tmp_path/'s'/'heldout_and_corruption.csv').open(encoding='utf-8') as f:
        rows = list(csv.DictReader(f))
    assert len(rows) == 114 * 3 * 2
    joint = [r for r in rows if r['method'] == 'joint_it2']
    assert all(r['fou_probability_lower'] and r['fou_probability_upper'] for r in joint)
    assert all(float(r['fou_probability_lower']) <= float(r['p_malignant'])
               <= float(r['fou_probability_upper']) for r in joint)
    assert all(r['blocked_reason'] for r in rows)


def test_seed_validation(tmp_path):
    with pytest.raises(ValueError,match='unique'):
        run(tmp_path,seeds=(17,17))
