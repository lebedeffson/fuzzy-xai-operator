import numpy as np

from fuzzyxai.risk.uncertainty import confidence_from_uncertainty, entropy_uncertainty, margin_uncertainty


def test_entropy_uncertainty_uniform_is_high():
    u = entropy_uncertainty([[0.5, 0.5]])[0]
    assert u > 0.99


def test_entropy_uncertainty_certain_is_low():
    u = entropy_uncertainty([[0.999, 0.001]])[0]
    assert u < 0.02


def test_margin_uncertainty_tracks_probability_gap():
    assert margin_uncertainty([[0.5, 0.5]])[0] > margin_uncertainty([[0.95, 0.05]])[0]


def test_confidence_from_uncertainty_clips():
    values = confidence_from_uncertainty(np.array([-1.0, 0.25, 2.0]))
    assert list(values) == [1.0, 0.75, 0.0]
