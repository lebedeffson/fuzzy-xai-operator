from __future__ import annotations

import runpy
from pathlib import Path

import pytest

from fuzzyxai import ExplainPlan, FuzzyXAI
from fuzzyxai.evidence import (
    CounterfactualEvidence,
    ExplanationEvidence,
    ExplanationGraph,
    SimilarCaseEvidence,
    build_explanation_claims,
    compose_human_explanation,
)
from fuzzyxai.schemas import validate_payload


class GroundwaterModel:
    classes_ = [0, 1]

    def predict_proba(self, rows):
        return [[0.18, 0.82] if row[0] > 0.5 else [0.78, 0.22] for row in rows]

    def predict(self, rows):
        return [1 if row[0] > 0.5 else 0 for row in rows]


def groundwater_plan() -> ExplainPlan:
    plan = ExplainPlan.default()
    plan.domain_language = {
        "features": {
            "fracture_density": {
                "label": "трещиноватость породы",
                "meaning": "количество и плотность трещин в горном массиве",
                "high_text": "Трещин больше, чем в большинстве исследованных участков.",
                "low_text": "Трещиноватость ниже типичного уровня.",
            },
            "water_saturation": {
                "label": "водонасыщенность",
                "high_text": "Водонасыщенность выше типичного уровня.",
            },
        },
        "classes": {0: {"label": "низкий риск"}, 1: {"label": "повышенный риск"}},
        "representations": {
            "normalized tabular feature vector": {"label": "значения геологических показателей"}
        },
        "actions": {
            "review": {
                "label": "Проверить специалистом",
                "explanation": "Передать результат специалисту и проверить ключевые геологические признаки.",
            }
        },
    }
    return plan


def test_domain_language_round_trip_and_validation() -> None:
    plan = groundwater_plan()
    restored = ExplainPlan.from_dict(plan.to_dict())
    assert restored.domain_language["features"]["fracture_density"]["label"] == "трещиноватость породы"
    payload = plan.to_dict()
    payload["domain_language"] = {"features": []}
    with pytest.raises(ValueError, match="domain_language.features"):
        ExplainPlan.from_dict(payload)


def test_domain_user_receives_small_grounded_cards_without_internal_terms() -> None:
    result = FuzzyXAI.wrap(GroundwaterModel(), explain_plan=groundwater_plan()).explain_one(
        [0.84, 0.72],
        object_id="85",
        feature_names=["fracture_density", "water_saturation"],
        # P17: two extra reference points close to the query were added so
        # the predicted class's concept is genuinely close to its prototype
        # (query_distance <= intra_class_variability) — class_concept no
        # longer defaults to "favorable" for the predicted class without a
        # real measured closeness (see evidence/concepts.py's query_row).
        reference_data=[[0.10, 0.22], [0.25, 0.31], [0.55, 0.61], [0.70, 0.68], [0.83, 0.71], [0.85, 0.73]],
        reference_labels=[0, 0, 1, 1, 1, 1],
        include_similar_cases=True,
        include_counterfactuals=True,
        evidence={
            "contributions": {"fracture_density": 0.62, "water_saturation": 0.18},
            "contribution_method": "native_controlled_score",
        },
    )
    explanation = result.explain_for()
    serialized = explanation.to_dict(include_technical_trace=False)
    assert validate_payload(serialized, "human_explanation").valid
    assert explanation.decision
    assert explanation.reliability
    assert explanation.recommended_action
    assert len(explanation.main_reasons) <= 3
    assert len(explanation.concerns) <= 2
    assert all(fragment.claim_refs and fragment.evidence_refs for fragment in explanation.fragments)
    assert "повышенный риск" in explanation.user_text
    assert "по сравнению" in explanation.user_text.lower() or "чем у" in explanation.user_text.lower()
    assert explanation.main_reasons[0].subject_label == "трещиноватость породы"
    assert all(reason.subject_label and reason.effect_direction and reason.comparison_text for reason in explanation.main_reasons)
    assert explanation.reliability.supported_by
    assert explanation.reliability.conclusion
    assert explanation.what_would_change_result
    change = explanation.what_would_change_result[0]
    assert change.feature == "fracture_density"
    assert change.original_value == pytest.approx(0.84)
    assert change.changed_value is not None
    assert change.direction == "decrease"
    assert change.prediction_before != change.prediction_after
    for forbidden in ("R31", "S4", "E5", "gamma", "rho", "claim_id", "defer_to_human", "audit_report"):
        assert forbidden not in explanation.user_text
    for vague in (
        "часть доступных сведений",
        "подтверждённая закономерность",
        "внутреннее представление модели",
        "нормализованные значения признаков",
        "проверенный контрфактический расчёт",
        "референсная выборка",
    ):
        assert vague not in explanation.user_text.lower()


def test_audience_and_detail_are_distinct_from_available_evidence_level() -> None:
    result = FuzzyXAI.wrap(GroundwaterModel(), explain_plan=groundwater_plan()).explain_one(
        [0.84, 0.72],
        object_id="85",
        feature_names=["fracture_density", "water_saturation"],
        reference_data=[[0.10, 0.22], [0.70, 0.68]],
    )
    assert result.explanation_level.startswith("E")
    assert result.explain_for("domain_user").audience == "domain_user"
    assert result.explain_for("ml_engineer").audience == "ml_engineer"
    assert "## Технические доказательства" in result.summary("ml_engineer", detail="full")
    with pytest.raises(ValueError, match="Russian"):
        result.explain_for(language="en")


def test_image_similarity_explains_what_the_percentage_means() -> None:
    similarity = SimilarCaseEvidence(
        query_object_id="image-85",
        reference_object_id="image-67",
        similarity_score=0.89,
        similarity_method="intersection_over_union",
        compared_representation="segmentation masks",
        matched_features=(),
        different_features=("surrounding_tissue_density",),
        matched_regions=("lesion_mask",),
        coverage_score=0.89,
        reference_label="research-class-B",
        reference_prediction="research-class-B",
        reference_outcome=None,
        limitations=("Research-only similarity; not a diagnosis probability.",),
        trace={"model_version": "medical-research-v1"},
        media_artifacts={"query_mask": "query-mask.png", "reference_mask": "reference-mask.png"},
    )
    evidence = ExplanationEvidence(similar_cases=(similarity,))
    claims = build_explanation_claims(
        evidence,
        prediction={"predictions": ["research-class-B"], "score": 0.76},
        diagnostics=(),
        action="review",
    )
    explanation = compose_human_explanation(
        claims,
        ExplanationGraph((), (), tuple(claims)),
        action="review",
        evidence=evidence,
        domain_language={
            "scope": "medical",
            "classes": {"research-class-B": {"label": "исследовательская группа B"}},
        },
    )
    text = explanation.user_text
    assert explanation.decision.domain_language_status == "insufficient_domain_language"
    assert "Предметное медицинское значение" in text
    assert "research-class-B" not in text
    assert "перекрываются на 89%" in text
    assert "геометрии выделенных областей" in text
    assert "не является вероятностью одинакового диагноза" in text


def test_incomplete_counterfactual_is_not_shown_to_domain_user() -> None:
    incomplete = CounterfactualEvidence(
        source_prediction=1,
        target_prediction=0,
        changed_features={"fracture_density": 0.61},
        changed_regions=(),
        changed_rules=(),
        minimality=None,
        plausibility=None,
        stability=None,
        expected_effect=None,
        observed_effect=None,
        actionability="requires domain review",
        limitations=("Original value was not recorded.",),
        evidence_refs=("counterfactual:test",),
    )
    evidence = ExplanationEvidence(counterfactuals=(incomplete,))
    claims = build_explanation_claims(
        evidence,
        prediction={"predictions": [1], "score": 0.82},
        diagnostics=(),
        action="review",
    )
    explanation = compose_human_explanation(
        claims,
        ExplanationGraph((), (), tuple(claims)),
        action="review",
        evidence=evidence,
        domain_language=groundwater_plan().domain_language,
    )
    assert explanation.what_would_change_result == ()


def test_direct_feature_reason_outranks_similar_case() -> None:
    result = FuzzyXAI.wrap(GroundwaterModel(), explain_plan=groundwater_plan()).explain_one(
        [0.84, 0.72],
        object_id="85",
        feature_names=["fracture_density", "water_saturation"],
        reference_data=[[0.10, 0.22], [0.25, 0.31], [0.55, 0.61], [0.70, 0.68]],
        reference_labels=[0, 0, 1, 1],
        include_similar_cases=True,
        evidence={
            "contributions": {"fracture_density": 0.62},
            "contribution_method": "native_controlled_score",
        },
    )
    explanation = result.explain_for()
    assert explanation.main_reasons[0].subject_label == "трещиноватость породы"
    assert explanation.main_reasons[0].effect_direction == "supports"


def test_comprehension_scorer_preserves_not_run_and_scores_complete_pilot() -> None:
    module = runpy.run_path(str(Path(__file__).parents[1] / "scripts/score_comprehension_pilot.py"))
    score_rows = module["score_rows"]
    assert score_rows([])["status"] == "planned_not_run"
    rows = []
    for index in range(6):
        role = "domain_specialist" if index < 3 else "model_integrator"
        for scenario in ("forgetting_case", "rule_ablation", "image_similarity"):
            for mode in ("technical_baseline", "human_explanation"):
                rows.append(
                    {
                    "participant_id": f"P{index + 1}",
                    "role": role,
                    "condition_order": "AB" if index % 2 == 0 else "BA",
                    "scenario_id": scenario,
                    "mode": mode,
                    "decision_correct": "true",
                    "reasons_correct": "true",
                    "concern_correct": "false" if mode == "technical_baseline" else "true",
                    "reliability_correct": "true",
                    "action_correct": "true",
                    "limitation_correct": "true",
                    "provenance_correct": "true",
                    "similarity_correct": "true",
                    "counterfactual_correct": "true",
                    "native_surrogate_correct": "true",
                    "overtrust_error": "false",
                    "iou_misinterpreted_as_probability": "false",
                    "sensitivity_misinterpreted_as_recommendation": "false",
                    "unsupported_inference_count": "0",
                    "completion_time_sec": "32" if mode == "technical_baseline" else "30",
                    "subjective_clarity_1_5": "4",
                    "cognitive_load_1_5": "2",
                    "notes": "controlled unit-test row",
                    }
                )
    scored = score_rows(rows)
    assert scored["status"] == "pass"
    assert scored["participant_count"] == 6
    assert scored["claim_allowed"] is True
