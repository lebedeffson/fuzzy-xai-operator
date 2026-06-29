from __future__ import annotations

import pytest

from fuzzyxai import HesitantFS, IntervalFS, NeutrosophicFS


def test_interval_delta_equals_half_width():
    fs = IntervalFS(lambda x: 0.2, lambda x: 0.8)
    _, delta = fs.reduce_to_f0()
    assert delta == pytest.approx(0.3)


def test_hesitant_delta_equals_max_deviation_from_mean():
    fs = HesitantFS(lambda x: [0.2, 0.5, 0.8])
    red, delta = fs.reduce_to_f0()
    mean = red.membership(0.5)
    assert delta == pytest.approx(max(abs(v - mean) for v in [0.2, 0.5, 0.8]))


def test_neutrosophic_delta_equals_distance_to_embedded_reduction():
    fs = NeutrosophicFS(lambda x: 0.78, lambda x: 0.20, lambda x: 0.64)
    red, delta = fs.reduce_to_f0()
    embedded = NeutrosophicFS(red.mu, lambda x: 0.0, lambda x: 1.0 - red.membership(x))
    assert delta == pytest.approx(fs.distance(embedded))


def test_reduction_acceptance_by_delta_threshold():
    fs = IntervalFS(lambda x: 0.45, lambda x: 0.55)
    _, delta = fs.reduce_to_f0()
    assert delta <= 0.05 + 1e-9


def test_reduction_rejected_when_delta_exceeds_threshold():
    fs = IntervalFS(lambda x: 0.1, lambda x: 0.9)
    _, delta = fs.reduce_to_f0()
    assert delta > 0.05
