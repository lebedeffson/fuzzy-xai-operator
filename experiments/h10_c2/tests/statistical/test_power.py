from __future__ import annotations

from h10_c2.statistics.power_simulation import approximate_paired_power, simulate_paired_power


BASE = {
    "pipelines": 6,
    "baseline_rate": 0.5,
    "alpha": 0.05,
    "icc": 0.02,
    "attrition": 0.05,
    "comparisons": 2,
}


def test_power_increases_with_effect() -> None:
    low = approximate_paired_power(cases_per_pipeline=200, effect=0.01, **BASE)
    high = approximate_paired_power(cases_per_pipeline=200, effect=0.08, **BASE)
    assert high > low


def test_power_increases_with_sample_size() -> None:
    low = approximate_paired_power(cases_per_pipeline=50, effect=0.05, **BASE)
    high = approximate_paired_power(cases_per_pipeline=500, effect=0.05, **BASE)
    assert high > low


def test_clustering_reduces_power() -> None:
    independent = approximate_paired_power(cases_per_pipeline=200, effect=0.05, **{**BASE, "icc": 0.0})
    clustered = approximate_paired_power(cases_per_pipeline=200, effect=0.05, **{**BASE, "icc": 0.2})
    assert independent > clustered


def test_more_comparisons_do_not_raise_power() -> None:
    few = approximate_paired_power(cases_per_pipeline=200, effect=0.05, **{**BASE, "comparisons": 1})
    many = approximate_paired_power(cases_per_pipeline=200, effect=0.05, **{**BASE, "comparisons": 4})
    assert few >= many


def test_simulated_power_is_reproducible() -> None:
    parameters = {**BASE, "cases_per_pipeline": 200, "effect": 0.05}
    first = simulate_paired_power(repetitions=2000, seed=44, **parameters)
    second = simulate_paired_power(repetitions=2000, seed=44, **parameters)
    assert first == second
