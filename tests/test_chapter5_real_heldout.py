from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np
import pytest

from experiments.chapter5_real_heldout import (
    STRESSORS, TARGET_ACCEPT_COVERAGE, _risk_components,
    calibrated_thresholds, corrupt, decisions, features_and_proba,
    one_split, prepare, run,
)


def test_split_is_disjoint_and_preprocessing_train_only() -> None:
    prepared = prepare(42)
    assert len(prepared.y_train) + len(prepared.y_val) + len(prepared.y_test) == 569
    assert len(prepared.y_test) == 114
    assert len(set(prepared.ids_test) & set(prepared.ids_train)) == 0
    assert len(set(prepared.ids_test) & set(prepared.ids_val)) == 0
    np.testing.assert_allclose(prepared.medians, np.median(prepared.x_train, axis=0))
    np.testing.assert_allclose(prepared.stdev, np.maximum(np.std(prepared.x_train, axis=0), 1e-9))


def test_corruptions_follow_predeclared_intensity_and_are_seed_stable() -> None:
    prepared = prepare(42)
    for kind in STRESSORS:
        a, m1 = corrupt(prepared.x_test, kind, np.random.default_rng(11), prepared)
        b, m2 = corrupt(prepared.x_test, kind, np.random.default_rng(11), prepared)
        np.testing.assert_allclose(a, b, equal_nan=True)
        np.testing.assert_array_equal(m1, m2)
        if kind == 'missing_40pct':
            assert np.all(m1.sum(axis=1) == 12)
        if kind == 'missing_10pct':
            assert np.all(m1.sum(axis=1) == 3)


def test_observer_and_equal_guardrail_never_accept_critical_cases() -> None:
    prepared = prepare(123)
    from sklearn.ensemble import RandomForestClassifier
    clf = RandomForestClassifier(n_estimators=20, random_state=123).fit(prepared.x_train, prepared.y_train)
    val = features_and_proba(clf, prepared.x_val, np.zeros_like(prepared.x_val, dtype=bool), prepared)
    thresholds = calibrated_thresholds(val)
    assert 0 < thresholds['theta_1'] < thresholds['theta_2'] < thresholds['theta_3'] < thresholds['theta_4'] < 1
    degraded, mask = corrupt(prepared.x_test, 'missing_40pct', np.random.default_rng(123), prepared)
    obs = features_and_proba(clf, degraded, mask, prepared)
    dec = decisions(obs, 0.7, thresholds)
    assert obs['critical'].all()
    assert not dec['observer_accept'].any()
    assert not dec['equal_guardrail_accept'].any()
    assert np.all(dec['actions'] == 'block')
    assert len(_risk_components(obs, 0)) == 5


def test_inconsistent_missingness_and_unknown_faults_rejected() -> None:
    prepared = prepare(42)
    with pytest.raises(ValueError, match='missing mask disagrees'):
        from sklearn.ensemble import RandomForestClassifier
        clf = RandomForestClassifier(n_estimators=5, random_state=42).fit(prepared.x_train, prepared.y_train)
        features_and_proba(clf, prepared.x_test, np.ones_like(prepared.x_test, dtype=bool), prepared)
    with pytest.raises(ValueError, match='unknown perturbation'):
        corrupt(prepared.x_test, 'invented_pathology', np.random.default_rng(1), prepared)


def test_real_heldout_evidence_and_reproducibility(tmp_path: Path) -> None:
    out_a, out_b = tmp_path / 'a', tmp_path / 'b'
    a = run(out_a, seeds=(42,), n_boot=100)
    b = run(out_b, seeds=(42,), n_boot=100)
    assert a == b
    assert (out_a / 'summary.json').read_bytes() == (out_b / 'summary.json').read_bytes()
    assert (out_a / 'heldout_cases.csv').read_bytes() == (out_b / 'heldout_cases.csv').read_bytes()
    assert len(a['splits']) == 1 and a['n_test_cases_total'] == 4 * 114
    assert a['clinical_metrics'] is None
    assert a['full_semantic_explanation_route_validated'] is False
    with (out_a / 'group_metrics.csv').open() as f:
        rows = list(csv.DictReader(f))
    assert len(rows) == len(STRESSORS) * 3
    severe = [r for r in rows if r['perturbation'] == 'missing_40pct']
    assert {r['policy'] for r in severe} == {'confidence_only','confidence_equal_guardrail','canonical_risk_with_qc'}
    assert next(r for r in severe if r['policy'] == 'canonical_risk_with_qc')['auto_accepted_with_critical_flag'] == '0'
    assert next(r for r in severe if r['policy'] == 'confidence_equal_guardrail')['auto_accepted_with_critical_flag'] == '0'
    clean = [r for r in rows if r['perturbation'] == 'clean']
    assert all(0.40 <= float(r['coverage']) <= 0.90 for r in clean)
    assert 0.0 <= a['splits'][0]['model_roc_auc_clean'] <= 1.0
    assert 0.0 < TARGET_ACCEPT_COVERAGE < 1.0
    assert len(json.loads((out_a / 'summary.json').read_text())['limitations']) >= 4
