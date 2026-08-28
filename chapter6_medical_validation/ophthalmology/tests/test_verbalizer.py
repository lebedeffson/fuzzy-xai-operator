from chapter6_medical_validation.ophthalmology.src.verbalizer import guard_strict_text, preservation_metrics


def test_strict_guard_rejects_clinical_claim_and_new_number():
    result = guard_strict_text("Пациент страдает. Риск 99.", allowed_claim_texts=[], forbidden_phrases=["пациент страдает"], allowed_numbers={"1"})
    assert result["accepted"] is False
    assert result["forbidden_phrases"]
    assert result["unsupported_numbers"] == ["99"]


def test_preservation_metrics_keep_action_and_numbers():
    metrics = preservation_metrics("Стадия 2. Действие block.", "Стадия 2. Действие block.", source_facts={"Стадия 2"}, action="block", limitations=[])
    assert metrics["P_fact"] == metrics["P_num"] == metrics["P_action"] == 1.0
    assert metrics["H"] == 0.0
