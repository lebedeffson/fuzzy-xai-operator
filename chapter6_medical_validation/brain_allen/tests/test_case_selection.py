from __future__ import annotations

import numpy as np

from chapter6_medical_validation.brain_allen.scripts.select_cases import choose, normalized_l1_disagreement


def test_map_disagreement_is_signed_support_diagnostic_not_gamma() -> None:
    assert normalized_l1_disagreement(np.asarray([[1.0, 0.0]]), np.asarray([[1.0, 0.0]])) == 0.0
    assert normalized_l1_disagreement(np.asarray([[1.0, 0.0]]), np.asarray([[0.0, 1.0]])) == 1.0


def test_selection_marks_absent_error_instead_of_inventing_one() -> None:
    rows = [
        {"object_id": "h", "correct": True, "label": 1, "confidence": 0.9, "top1_top2_margin": 0.8, "xai_diagnostic_disagreement": 0.1, "technical_quality_score": 0.9},
        {"object_id": "o", "correct": True, "label": 0, "confidence": 0.8, "top1_top2_margin": 0.6, "xai_diagnostic_disagreement": 0.2, "technical_quality_score": 0.7},
    ]
    selected = choose(rows)
    assert selected["BRAIN_D"]["status"] == "not_available"
    assert selected["BRAIN_E"]["object_id"] == "o"
