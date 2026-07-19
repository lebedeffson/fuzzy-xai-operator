"""Build deterministic golden explanations for the v1.1 experience contract."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

from fuzzyxai import FuzzyXAI
from fuzzyxai.adapters import NativeRuleAdapter
from fuzzyxai.evidence import ExplanationClaim, ExplanationEvidence, SimilarCaseEvidence
from fuzzyxai.evidence import evaluate_rule_ablation, extract_rules


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "release_evidence/explanation_experience"


class ControlledRuleModel:
    classes_ = [0, 1]
    rules_ = [
        {
            "rule_id": "R31",
            "antecedents": ["fracture_density is high", "distance is low"],
            "consequent": "1",
            "activation": 0.71,
            "coverage": 0.08,
            "precision": 0.91,
            "support": 8,
            "stability": 0.42,
            "importance": 0.84,
        },
        {
            "rule_id": "R12",
            "antecedents": ["water_saturation is high"],
            "consequent": "1",
            "activation": 0.82,
            "coverage": 0.46,
            "precision": 0.86,
            "support": 46,
            "stability": 0.88,
            "importance": 0.67,
        },
    ]

    def predict_proba(self, values):
        return [[0.18, 0.82] if row[0] >= 0.5 else [0.78, 0.22] for row in values]

    def predict(self, values):
        return [1 if row[0] >= 0.5 else 0 for row in values]


def stable_payload(result) -> dict:
    payload = result.to_dict()
    payload["trace"]["generated_at"] = "controlled_fixture"
    return payload


def write_json(path: Path, payload: object) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def build_object_85() -> dict:
    model = ControlledRuleModel()
    fx = FuzzyXAI.wrap(model)
    rule = next(rule for rule in extract_rules(NativeRuleAdapter(model), feature_names=["fracture_density", "distance"]) if rule.rule_id == "R31")
    rule = evaluate_rule_ablation(
        rule,
        baseline_metrics={"train": 0.86, "validation": 0.82, "test": 0.84},
        ablated_metrics={"train": 0.85, "validation": 0.75, "test": 0.79},
    )
    training = fx.observe_training(
        history={
            "objects": {
                "85": [
                    {"epoch": 7, "correct": True, "confidence": 0.78, "loss": 0.24, "margin": 0.42, "rule_activations": {"R31": 0.71}},
                    {"epoch": 16, "correct": False, "confidence": 0.31, "loss": 0.79, "margin": -0.16, "rule_activations": {"R31": 0.08}},
                ]
            },
            "global_metric": [0.80, 0.84],
            "subgroup_metrics": {"S4": [0.72, 0.49]},
            "subgroup_objects": {"S4": ["85"]},
            "subgroup_rule_history": {"S4": [["R31"], []]},
        }
    )
    result = fx.explain_one(
        [0.84, 0.2],
        object_id="85",
        feature_names=["fracture_density", "distance"],
        reference_data=[[0.1, 0.8], [0.2, 0.7], [0.6, 0.3], [0.7, 0.2]],
        reference_ids=["11", "12", "67", "68"],
        reference_labels=[0, 0, 1, 1],
        training_run=training,
        include_training_trace=True,
        include_similar_cases=True,
        include_counterfactuals=True,
        include_model_knowledge=False,
        additional_evidence=ExplanationEvidence(rules=[rule]),
        dataset_version="controlled_object85_v1",
        evidence={
            "alignment": {"components": {"rules": 0.10}, "weights": {"rules": 1.0}},
            "reduction": {"components": {"rules": 0.12}, "weights": {"rules": 1.0}},
            "risk": {
                "components": {"forgetting": 0.62},
                "weights": {"forgetting": 1.0},
                "thresholds": {"theta_1": 0.2, "theta_2": 0.4, "theta_3": 0.6, "theta_4": 0.8},
            },
        },
    )
    write_json(OUTPUT / "object_85_explanation.json", stable_payload(result))
    (OUTPUT / "object_85_overview.md").write_text(result.overview() + "\n" + result.story(), encoding="utf-8")
    result.visualize(view="explanation_story", output=OUTPUT / "object_85_story.png")
    result.visualize(view="training_trace", output=OUTPUT / "object_85_training_trace.png")
    result.inspect("rule:R31").visualize(view="rule_ablation", output=OUTPUT / "object_85_rule_ablation.png")
    return {"level": result.explanation_level, "action": result.action, "claim_count": len(result.claims)}


def build_anfis() -> dict:
    result = FuzzyXAI.wrap(ControlledRuleModel()).explain_one(
        [0.76, 0.24],
        object_id="anfis-fixture-1",
        feature_names=["fracture_density", "distance"],
        reference_data=[[0.1, 0.8], [0.2, 0.7], [0.6, 0.3], [0.7, 0.2]],
        reference_labels=[0, 0, 1, 1],
        include_similar_cases=True,
        include_model_knowledge=True,
        dataset_version="controlled_anfis_v1",
    )
    write_json(OUTPUT / "anfis_native_rules_explanation.json", stable_payload(result))
    write_json(OUTPUT / "anfis_decision_evidence.json", result.view_model.visual_spec["decision_evidence"])
    result.visualize(view="knowledge_atlas", output=OUTPUT / "anfis_knowledge_atlas.png")
    return {"level": result.explanation_level, "action": result.action, "claim_count": len(result.claims)}


def build_medical_research_fixture() -> dict:
    cases = [
        SimilarCaseEvidence(
            query_object_id="research_image_query",
            reference_object_id="67",
            similarity_score=0.89,
            similarity_method="intersection_over_union",
            compared_representation="segmentation masks",
            matched_features=(),
            different_features=("surrounding_tissue_density",),
            matched_regions=("segmented_region",),
            coverage_score=0.89,
            reference_label="research_class_B",
            reference_prediction="research_class_B",
            reference_outcome=None,
            limitations=("Mask overlap is not the probability of an identical diagnosis.",),
            trace={"fixture": "controlled_medical_similarity_v1"},
        ),
        SimilarCaseEvidence(
            query_object_id="research_image_query",
            reference_object_id="67",
            similarity_score=0.82,
            similarity_method="cosine_similarity",
            compared_representation="model embedding vector",
            matched_features=("boundary_shape",),
            different_features=("surrounding_tissue_density",),
            matched_regions=(),
            coverage_score=None,
            reference_label="research_class_B",
            reference_prediction="research_class_B",
            reference_outcome=None,
            limitations=("Embedding similarity is model-dependent and is not a clinical conclusion.",),
            trace={"fixture": "controlled_medical_similarity_v1"},
        ),
    ]
    claims = [
        ExplanationClaim(
            claim_id="C-MED-001",
            claim_type="similar_case",
            scope="object",
            subject_id="research_image_query",
            statement="Пересечение сегментационных масок объекта и примера 67 по IoU равно 0.89.",
            short_statement="Mask IoU with object 67: 0.89",
            status="supported",
            strength=0.89,
            evidence_refs=("similar:research_image_query:67:iou",),
            limitations=("IoU характеризует геометрию масок, а не вероятность диагноза.",),
            metric_name="mask_iou",
            metric_value=0.89,
        ),
        ExplanationClaim(
            claim_id="C-MED-002",
            claim_type="similar_case",
            scope="object",
            subject_id="research_image_query",
            statement="Косинусное сходство embedding объекта и примера 67 равно 0.82.",
            short_statement="Embedding similarity with object 67: 0.82",
            status="supported",
            strength=0.82,
            evidence_refs=("similar:research_image_query:67:embedding",),
            limitations=("Метрика зависит от версии модели и не является клиническим выводом.",),
            metric_name="embedding_cosine_similarity",
            metric_value=0.82,
        ),
    ]
    payload = {
        "status": "controlled_fixture_research_only",
        "clinical_claims": False,
        "similar_cases": [case.to_dict() for case in cases],
        "claims": [claim.to_dict() for claim in claims],
        "required_user_message": "Similarity supports inspection but is not a probability of the same diagnosis.",
    }
    write_json(OUTPUT / "medical_research_similarity_explanation.json", payload)
    return {"status": payload["status"], "claim_count": len(claims)}


def main() -> None:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    summary = {
        "schema_version": "1.0",
        "status": "controlled_golden_explanations",
        "object_85": build_object_85(),
        "anfis": build_anfis(),
        "medical_research": build_medical_research_fixture(),
        "comprehension_study": "planned_not_run",
    }
    write_json(OUTPUT / "golden_summary.json", summary)
    files = {}
    for path in sorted(OUTPUT.iterdir()):
        if path.name == "manifest_sha256.json" or path.suffix == ".html" or not path.is_file():
            continue
        files[path.name] = hashlib.sha256(path.read_bytes()).hexdigest()
    write_json(OUTPUT / "manifest_sha256.json", {"schema_version": "1.0", "files": files})
    print("PASS: object_85_golden")
    print("PASS: anfis_golden")
    print("PASS: medical_research_fixture")
    print("PASS: explanation_experience_manifest")


if __name__ == "__main__":
    main()
