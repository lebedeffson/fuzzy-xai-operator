import json

from fuzzyxai.experiments.h10_c4 import _load_or_create_calibration


def test_development_calibration_is_reused_without_held_out_data(
    tmp_path, development_scenarios
) -> None:
    path = tmp_path / "calibration.json"
    first = _load_or_create_calibration(development_scenarios, path)
    payload = json.loads(path.read_text())
    second = _load_or_create_calibration(development_scenarios, path)

    assert first == second
    assert payload["fitted_split"] == "development"
    assert payload["held_out_used_for_calibration"] is False
    assert payload["status"] == "FROZEN_BEFORE_HELD_OUT_EXECUTION"
