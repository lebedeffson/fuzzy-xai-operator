from __future__ import annotations

import pytest

from fuzzyxai import ExplainPlan, FuzzyXAI
from fuzzyxai.evidence import (
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
        reference_data=[[0.10, 0.22], [0.25, 0.31], [0.55, 0.61], [0.70, 0.68]],
        reference_labels=[0, 0, 1, 1],
        include_similar_cases=True,
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
    for forbidden in ("R31", "S4", "E5", "gamma", "rho", "claim_id", "defer_to_human", "audit_report"):
        assert forbidden not in explanation.user_text


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
    )
    text = explanation.user_text
    assert "перекрываются на 89%" in text
    assert "геометрии выделенных областей" in text
    assert "не является вероятностью одинакового диагноза" in text
