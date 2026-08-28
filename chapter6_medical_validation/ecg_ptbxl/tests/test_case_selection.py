from __future__ import annotations

import numpy as np

from chapter6_medical_validation.ecg_ptbxl.scripts.run_cases import diagnostic_disagreement, select_initial_cases, signed_correlation


def test_ecg_selection_is_deterministic_and_missing_errors_are_explicit() -> None:
    rows = [
        {"ecg_id": 1, "label": 0, "prediction": 0, "correct": True, "confidence": 0.9, "p_abnormal": 0.1, "technical_quality_score": 0.8},
        {"ecg_id": 2, "label": 1, "prediction": 1, "correct": True, "confidence": 0.8, "p_abnormal": 0.8, "technical_quality_score": 0.6},
    ]
    selected = select_initial_cases(rows)
    assert selected["ECG_A"]["ecg_id"] == 1
    assert selected["ECG_B"]["ecg_id"] == 2
    assert selected["ECG_D"]["status"] == "not_available"
    assert selected["ECG_G"]["ecg_id"] == 2


def test_common_grid_distance_is_not_renamed_gamma() -> None:
    assert diagnostic_disagreement(np.zeros((12, 20)), np.zeros((12, 20))) == 0.0
    left = np.zeros((12, 20)); right = np.zeros((12, 20)); left[0, 0] = 1.0; right[0, 1] = 1.0
    assert diagnostic_disagreement(left, right) == 1.0


def test_constant_common_grid_correlation_is_explicitly_not_applicable() -> None:
    result = signed_correlation(np.zeros((12, 20)), np.ones((12, 20)))
    assert result["value"] is None
    assert result["status"] == "not_applicable"
