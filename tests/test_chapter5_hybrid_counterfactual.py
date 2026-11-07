from __future__ import annotations

from experiments.chapter5_hybrid_counterfactual import controlled_cases, predeclared_conflict, run_controlled_grid


def test_case_grid_is_not_a_fixed_outcome_sequence() -> None:
    cases = controlled_cases()
    assert len(cases) == 64
    assert len({(c['image_quality'], c['segmentation_quality'], c['model_match_signal'], c['alpha_accept']) for c in cases}) == 64
    assert 0 < sum(predeclared_conflict(c) for c in cases) < 64


def test_computed_routes_preserve_controlled_guardrail(tmp_path) -> None:
    report = run_controlled_grid(tmp_path)
    assert report['n_cases'] == 64
    assert report['fuzzyxai_missed_guardrail'] == 0
    assert report['proofs_verified'] == 64
    assert report['clinical_or_independent_biometric_validation'] is False
    assert (tmp_path / 'cases.csv').is_file()
