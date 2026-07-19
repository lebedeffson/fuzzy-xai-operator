"""Build deterministic golden explanations for the v1.2 human experience contract."""

from __future__ import annotations

import hashlib
import json
from dataclasses import replace
from pathlib import Path

import numpy as np

from fuzzyxai import ExplainPlan, FuzzyXAI
from fuzzyxai.adapters import NativeRuleAdapter
from fuzzyxai.evidence import (
    ExplanationClaim,
    ExplanationEvidence,
    ExplanationGraph,
    SimilarCaseEvidence,
    build_explanation_claims,
    compare_region_masks,
    compose_human_explanation,
)
from fuzzyxai.evidence import evaluate_rule_ablation, extract_rules


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "release_evidence/explanation_experience"


def groundwater_plan() -> ExplainPlan:
    plan = ExplainPlan.default()
    plan.domain_language = {
        "features": {
            "fracture_density": {
                "label": "трещиноватость породы",
                "meaning": "количество и плотность трещин в горном массиве",
                "high_text": "Трещин больше, чем в большинстве исследованных участков.",
            },
            "water_saturation": {
                "label": "водонасыщенность",
                "high_text": "Водонасыщенность выше типичного уровня.",
            },
            "distance": {"label": "расстояние до выработки"},
            "pressure": {"label": "давление воды"},
        },
        "classes": {"0": {"label": "низкий риск"}, "1": {"label": "повышенный риск"}},
        "actions": {
            "review": {
                "label": "Проверить специалистом",
                "explanation": "Передать результат специалисту и отдельно проверить трещиноватость и водонасыщенность.",
            },
            "defer_to_human": {
                "label": "Передать специалисту",
                "explanation": "Передать результат специалисту и не применять его автоматически до предметной проверки.",
            },
        },
    }
    return plan


def object_85_history() -> list[dict[str, object]]:
    """Controlled 12-checkpoint trajectory with one observable forgetting event."""

    epochs = (1, 4, 7, 10, 13, 16, 19, 22, 25, 28, 31, 34)
    confidence = (0.32, 0.46, 0.72, 0.78, 0.75, 0.31, 0.28, 0.30, 0.34, 0.37, 0.39, 0.41)
    loss = (0.91, 0.62, 0.31, 0.24, 0.28, 0.79, 0.83, 0.78, 0.71, 0.68, 0.64, 0.61)
    margin = (-0.24, -0.08, 0.21, 0.42, 0.35, -0.16, -0.22, -0.18, -0.11, -0.07, -0.04, -0.02)
    r31 = (0.19, 0.38, 0.66, 0.71, 0.63, 0.08, 0.06, 0.05, 0.04, 0.04, 0.03, 0.03)
    r12 = (0.31, 0.45, 0.56, 0.62, 0.68, 0.73, 0.77, 0.79, 0.81, 0.82, 0.83, 0.84)
    global_metric = (0.75, 0.77, 0.79, 0.80, 0.81, 0.82, 0.83, 0.835, 0.84, 0.845, 0.85, 0.855)
    subgroup_metric = (0.70, 0.71, 0.72, 0.72, 0.69, 0.49, 0.47, 0.46, 0.48, 0.50, 0.51, 0.52)
    return [
        {
            "epoch": epoch,
            "correct": 7 <= epoch < 16,
            "predicted_class": 1 if 7 <= epoch < 16 else 0,
            "confidence": confidence[index],
            "loss": loss[index],
            "margin": margin[index],
            "prototype_distance": round(0.61 - 0.018 * index if epoch < 16 else 0.71 + 0.009 * (index - 5), 3),
            "global_metric": global_metric[index],
            "subgroup_metric": subgroup_metric[index],
            "rule_activations": {"R31": r31[index], "R12": r12[index]},
        }
        for index, epoch in enumerate(epochs)
    ]


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
    fx = FuzzyXAI.wrap(model, explain_plan=groundwater_plan())
    rule = next(rule for rule in extract_rules(NativeRuleAdapter(model), feature_names=["fracture_density", "distance"]) if rule.rule_id == "R31")
    rule = evaluate_rule_ablation(
        rule,
        baseline_metrics={"train": 0.86, "validation": 0.82, "test": 0.84, "subgroup_recall": 0.72, "critical_errors": 2.0, "calibration": 0.11},
        ablated_metrics={"train": 0.85, "validation": 0.75, "test": 0.79, "subgroup_recall": 0.49, "critical_errors": 7.0, "calibration": 0.24},
    )
    history = object_85_history()
    training = fx.observe_training(
        history={
            "objects": {"85": history},
            "global_metric": [float(item["global_metric"]) for item in history],
            "subgroup_metrics": {"S4": [float(item["subgroup_metric"]) for item in history]},
            "subgroup_objects": {"S4": ["85"]},
            "subgroup_rule_history": {"S4": [["R31"] if int(item["epoch"]) < 16 else [] for item in history]},
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
    human = result.explain_for("domain_user")
    write_json(OUTPUT / "object_85_human_explanation.json", human.to_dict(include_technical_trace=False))
    (OUTPUT / "object_85_human_explanation.md").write_text(human.user_text, encoding="utf-8")
    (OUTPUT / "object_85_overview.md").write_text(result.overview() + "\n" + result.story(), encoding="utf-8")
    result.visualize(view="explanation_story", output=OUTPUT / "object_85_story.png")
    result.visualize(view="training_trace", output=OUTPUT / "object_85_training_trace.png")
    result.inspect("rule:R31").visualize(view="rule_ablation", output=OUTPUT / "object_85_rule_ablation.png")
    assert len(result.view_model.visual_spec["training_timeline"][0]["points"]) >= 12
    assert result.view_model.visual_spec["audit"]["graph_valid"] is True
    return {"level": result.explanation_level, "action": result.action, "claim_count": len(result.claims)}


def build_anfis() -> dict:
    result = FuzzyXAI.wrap(ControlledRuleModel(), explain_plan=groundwater_plan()).explain_one(
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


def build_cross_model_matrix() -> dict:
    """Exercise independent black-box, linear sklearn, and tree channels."""

    from sklearn.datasets import make_classification
    from sklearn.linear_model import LogisticRegression
    from sklearn.tree import DecisionTreeClassifier

    values, labels = make_classification(
        n_samples=96,
        n_features=4,
        n_informative=3,
        n_redundant=0,
        random_state=85,
    )
    feature_names = ["fracture_density", "water_saturation", "distance", "pressure"]

    class BlackBox:
        def __call__(self, rows):
            return [int(float(row[0]) + float(row[1]) > 0.0) for row in rows]

    models = {
        "black_box_callable": BlackBox(),
        "sklearn_linear": LogisticRegression(max_iter=500, random_state=85).fit(values, labels),
        "tree_native_paths": DecisionTreeClassifier(max_depth=4, random_state=85).fit(values, labels),
    }
    output_dir = OUTPUT / "cross_model"
    output_dir.mkdir(parents=True, exist_ok=True)
    matrix: dict[str, object] = {}
    for scenario_id, model in models.items():
        result = FuzzyXAI.wrap(model, explain_plan=groundwater_plan()).explain_one(
            values[0],
            object_id=f"{scenario_id}:0",
            feature_names=feature_names,
            reference_data=values,
            reference_ids=[f"train:{index}" for index in range(len(values))],
            reference_labels=labels.tolist(),
            include_similar_cases=True,
            include_counterfactuals=True,
            include_model_knowledge=True,
            dataset_version="controlled_cross_model_v1",
        )
        payload = stable_payload(result)
        write_json(output_dir / f"{scenario_id}.json", payload)
        result.visualize(view="explanation_story", output=output_dir / f"{scenario_id}_story.png")
        capabilities = result.view_model.trace["adapter_capabilities"]
        matrix[scenario_id] = {
            "adapter_id": result.prediction.adapter_id,
            "explanation_level": result.explanation_level,
            "native_channels": list(result.native_channels),
            "surrogate_channels": list(result.surrogate_channels),
            "missing_channels": list(result.missing_channels),
            "capabilities": capabilities,
            "graph_valid": result.explanation_graph.validate_reachability() == (),
            "action": result.action,
        }
    write_json(output_dir / "cross_model_matrix.json", matrix)
    assert matrix["black_box_callable"]["explanation_level"] in {"E1", "E3"}
    assert "rules" in matrix["sklearn_linear"]["surrogate_channels"]
    assert "rules" in matrix["tree_native_paths"]["native_channels"]
    return matrix


def build_medical_research_fixture() -> dict:
    import matplotlib.pyplot as plt

    media = OUTPUT / "medical_media"
    media.mkdir(parents=True, exist_ok=True)
    yy, xx = np.ogrid[:64, :64]
    query_mask = (xx - 31) ** 2 / 15**2 + (yy - 30) ** 2 / 11**2 <= 1
    reference_mask = query_mask.copy()
    inside = np.argwhere(query_mask)
    outside = np.argwhere(~query_mask)
    reference_mask[tuple(inside[:30].T)] = False
    reference_mask[tuple(outside[:30].T)] = True
    overlap = np.logical_and(query_mask, reference_mask)
    difference = np.logical_xor(query_mask, reference_mask)
    counter_mask_a = (xx - 22) ** 2 / 8**2 + (yy - 37) ** 2 / 17**2 <= 1
    counter_mask_b = np.logical_and((xx - 38) ** 2 + (yy - 27) ** 2 <= 13**2, yy < 34)
    base = np.linspace(0.15, 0.65, 64)[None, :] + np.linspace(0.0, 0.2, 64)[:, None]
    query_image = np.clip(base + 0.28 * query_mask, 0, 1)
    reference_image = np.clip(base + 0.25 * reference_mask + 0.02 * np.sin(xx / 3), 0, 1)
    counter_image_a = np.clip(base + 0.24 * counter_mask_a, 0, 1)
    counter_image_b = np.clip(base + 0.22 * counter_mask_b, 0, 1)
    assets = {
        "query_image": query_image,
        "reference_image_67": reference_image,
        "query_mask": query_mask,
        "reference_mask_67": reference_mask,
        "mask_intersection": overlap,
        "mask_difference": difference,
        "counterexample_41": counter_image_a,
        "counterexample_93": counter_image_b,
    }
    artifact_paths: dict[str, str] = {}
    for name, array in assets.items():
        path = media / f"{name}.png"
        plt.imsave(path, array, cmap="gray", vmin=0, vmax=1)
        artifact_paths[name] = str(path.relative_to(ROOT))

    measured_iou = compare_region_masks(
        query_mask,
        reference_mask,
        query_object_id="research_image_query",
        reference_object_id="67",
    )
    measured_iou = replace(
        measured_iou,
        reference_label="research_class_B",
        reference_prediction="research_class_B",
        reference_outcome="controlled_fixture",
        media_artifacts={key: artifact_paths[key] for key in ("query_image", "reference_image_67", "query_mask", "reference_mask_67", "mask_intersection", "mask_difference")},
    )
    cases = [
        measured_iou,
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
            media_artifacts={"query_image": artifact_paths["query_image"], "reference_image": artifact_paths["reference_image_67"]},
        ),
        SimilarCaseEvidence(
            query_object_id="research_image_query", reference_object_id="41", similarity_score=0.76,
            similarity_method="cosine_similarity", compared_representation="model embedding vector",
            matched_features=("boundary_shape",), different_features=("texture", "surrounding_tissue_density"), matched_regions=(), coverage_score=None,
            reference_label="research_class_A", reference_prediction="research_class_A", reference_outcome="controlled_fixture",
            limitations=("Counterexample similarity is representation-specific and research-only.",), trace={"fixture": "controlled_medical_similarity_v1"},
            is_counterexample=True, media_artifacts={"query_image": artifact_paths["query_image"], "reference_image": artifact_paths["counterexample_41"]},
        ),
        SimilarCaseEvidence(
            query_object_id="research_image_query", reference_object_id="93", similarity_score=0.71,
            similarity_method="cosine_similarity", compared_representation="model embedding vector",
            matched_features=("region_area",), different_features=("boundary_shape", "texture"), matched_regions=(), coverage_score=None,
            reference_label="research_class_A", reference_prediction="research_class_A", reference_outcome="controlled_fixture",
            limitations=("Counterexample similarity is representation-specific and research-only.",), trace={"fixture": "controlled_medical_similarity_v1"},
            is_counterexample=True, media_artifacts={"query_image": artifact_paths["query_image"], "reference_image": artifact_paths["counterexample_93"]},
        ),
    ]
    claims = [
        ExplanationClaim(
            claim_id="C-MED-001",
            claim_type="similar_case",
            scope="medical",
            subject_id="research_image_query",
            statement=f"Пересечение сегментационных масок объекта и примера 67 по IoU равно {measured_iou.similarity_score:.3f}.",
            short_statement=f"Mask IoU with object 67: {measured_iou.similarity_score:.3f}",
            evidence_status="supported",
            effect="neutral",
            severity="info",
            strength=measured_iou.similarity_score,
            evidence_refs=("similar:research_image_query:67:iou",),
            limitations=("IoU характеризует геометрию масок, а не вероятность диагноза.",),
            metric_name="mask_iou",
            metric_value=measured_iou.similarity_score,
            applicability="research_only",
        ),
        ExplanationClaim(
            claim_id="C-MED-002",
            claim_type="similar_case",
            scope="medical",
            subject_id="research_image_query",
            statement="Косинусное сходство embedding объекта и примера 67 равно 0.82.",
            short_statement="Embedding similarity with object 67: 0.82",
            evidence_status="supported",
            effect="neutral",
            severity="info",
            strength=0.82,
            evidence_refs=("similar:research_image_query:67:embedding",),
            limitations=("Метрика зависит от версии модели и не является клиническим выводом.",),
            metric_name="embedding_cosine_similarity",
            metric_value=0.82,
            applicability="research_only",
        ),
        ExplanationClaim(
            claim_id="C-MED-003", claim_type="counterexample", scope="medical", subject_id="research_image_query",
            statement="Два ближайших контролируемых контрпримера имеют сходные отдельные признаки, но относятся к research_class_A.",
            short_statement="Two nearest counterexamples limit automatic interpretation",
            evidence_status="supported", effect="adverse", severity="warning", strength=None,
            evidence_refs=("similar:research_image_query:41", "similar:research_image_query:93"),
            limitations=("Контрпримеры не заменяют клиническую валидацию.",), applicability="research_only",
        ),
    ]
    human_evidence = ExplanationEvidence(similar_cases=cases)
    human_claims = build_explanation_claims(
        human_evidence,
        prediction={"predictions": ["research_class_B"], "score": 0.76},
        diagnostics=(),
        action="review",
    )
    human = compose_human_explanation(
        human_claims,
        ExplanationGraph((), (), tuple(human_claims)),
        action="review",
        evidence=human_evidence,
        domain_language={
            "classes": {"research_class_B": {"label": "исследовательский класс B"}},
            "actions": {
                "review": {
                    "label": "Проверить специалистом",
                    "explanation": "Использовать сходство только как исследовательскую подсказку, а не как клинический вывод.",
                }
            },
        },
    )
    payload = {
        "status": "controlled_fixture_research_only",
        "clinical_claims": False,
        "similar_cases": [case.to_dict() for case in cases],
        "claims": [claim.to_dict() for claim in claims],
        "media_artifacts": artifact_paths,
        "mask_iou": measured_iou.similarity_score,
        "counterexample_count": 2,
        "required_user_message": "Similarity supports inspection but is not a probability of the same diagnosis.",
        "human_explanation": human.to_dict(include_technical_trace=False),
    }
    write_json(OUTPUT / "medical_research_similarity_explanation.json", payload)
    write_json(OUTPUT / "medical_research_human_explanation.json", human.to_dict(include_technical_trace=False))
    (OUTPUT / "medical_research_human_explanation.md").write_text(human.user_text, encoding="utf-8")
    return {"status": payload["status"], "claim_count": len(claims)}


def main() -> None:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    summary = {
        "schema_version": "1.2",
        "status": "controlled_golden_explanations",
        "object_85": build_object_85(),
        "anfis": build_anfis(),
        "medical_research": build_medical_research_fixture(),
        "cross_model": build_cross_model_matrix(),
        "comprehension_study": "planned_not_run",
    }
    write_json(OUTPUT / "golden_summary.json", summary)
    files = {}
    for path in sorted(OUTPUT.rglob("*")):
        if path.name == "manifest_sha256.json" or path.suffix == ".html" or not path.is_file():
            continue
        files[str(path.relative_to(OUTPUT))] = hashlib.sha256(path.read_bytes()).hexdigest()
    write_json(OUTPUT / "manifest_sha256.json", {"schema_version": "1.0", "files": files})
    print("PASS: object_85_golden")
    print("PASS: anfis_golden")
    print("PASS: medical_research_fixture")
    print("PASS: explanation_experience_manifest")


if __name__ == "__main__":
    main()
