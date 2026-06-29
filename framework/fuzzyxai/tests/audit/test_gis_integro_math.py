import pytest

from fuzzyxai.core.alignment import compute_gamma_route


def test_gis_gamma_route_uses_model_to_explanations_formula() -> None:
    assert compute_gamma_route(0.67, 0.72, 0.47) == pytest.approx(max(abs(0.67 - 0.72), abs(0.67 - 0.47)))
