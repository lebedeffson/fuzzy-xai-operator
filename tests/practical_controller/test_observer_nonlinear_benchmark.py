"""Developmental nonlinear Shapley benchmark: reproducibility and baselines."""

from __future__ import annotations

from experiments.observer_nonlinear_shap_benchmark import run_benchmark


def test_nonlinear_real_model_baseline_comparison_and_missing_evidence() -> None:
    report = run_benchmark(n_per_dataset=2, seed=2026)
    assert report["n_heldout_objects"] == 6
    assert report["n_fault_variants"] == 6 * 6
    assert report["maximum_clean_native_vs_oracle_disagreement"] < 1e-7
    cases = report["families"]
    assert cases["clean"]["certified"] == 6
    assert cases["clean"]["strong_sum_baseline_accepted"] == 6
    for family in ("zero_sum_0.02", "zero_sum_0.2"):
        assert cases[family]["blocked"] == 6
        assert cases[family]["strong_sum_baseline_accepted"] == 6
    assert cases["wrong_background"]["blocked"] == 6
    assert cases["wrong_background"]["strong_sum_baseline_accepted"] == 6
    assert cases["missing_background"]["review"] == 6
    assert cases["missing_background"]["strong_sum_baseline_accepted"] == 6
    assert cases["stale_input"]["blocked"] == 6
    assert cases["stale_input"]["strong_sum_baseline_accepted"] == 0
    assert report["claim_allowed"] is False
