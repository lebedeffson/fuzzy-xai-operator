from chapter6_medical_validation.ophthalmology.src.case_selection import select_registered_cases


def test_registered_case_selection_is_algorithmic_and_deterministic():
    rows = [
        {"sample_id": "a", "label": 0, "prediction": 0, "confidence": 0.9, "top1_top2_margin": 0.8, "technical_quality_score": 0.9, "explanation_disagreement": 0.1},
        {"sample_id": "b", "label": 2, "prediction": 2, "confidence": 0.8, "top1_top2_margin": 0.1, "technical_quality_score": 0.8, "explanation_disagreement": 0.2},
        {"sample_id": "c", "label": 1, "prediction": 4, "confidence": 0.95, "top1_top2_margin": 0.7, "technical_quality_score": 0.2, "explanation_disagreement": 0.8},
    ]
    selected = select_registered_cases(rows)
    assert selected["A_confident_correct_grade0"]["sample_id"] == "a"
    assert selected["B_confident_correct_referable"]["sample_id"] == "b"
    assert selected["C_boundary"]["sample_id"] == "b"
    assert selected["D_high_confidence_error"]["sample_id"] == "c"
    assert selected["E_lowest_technical_quality"]["sample_id"] == "c"
    assert selected["F_highest_explanation_disagreement"]["sample_id"] == "c"
