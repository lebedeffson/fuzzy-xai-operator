from __future__ import annotations

import re

from fuzzyxai import FuzzyXAI
from fuzzyxai.adapters import NativeRuleAdapter
from fuzzyxai.evidence import ExplanationEvidence, evaluate_rule_ablation, extract_rules


class TraceableRuleModel:
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
        }
    ]

    def predict_proba(self, values):
        return [[0.18, 0.82] if row[0] >= 0.5 else [0.78, 0.22] for row in values]

    def predict(self, values):
        return [1 if row[0] >= 0.5 else 0 for row in values]


def _full_result():
    model = TraceableRuleModel()
    fx = FuzzyXAI.wrap(model)
    rule = extract_rules(NativeRuleAdapter(model), feature_names=["fracture_density", "distance"])[0]
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
            }
        }
    )
    return fx.explain_one(
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
        evidence={
            "alignment": {"components": {"rules": 0.1}, "weights": {"rules": 1.0}},
            "reduction": {"components": {"rules": 0.12}, "weights": {"rules": 1.0}},
            "risk": {
                "components": {"forgetting": 0.62},
                "weights": {"forgetting": 1.0},
                "thresholds": {"theta_1": 0.2, "theta_2": 0.4, "theta_3": 0.6, "theta_4": 0.8},
            },
        },
    )


def test_claims_are_grounded_and_primary_graph_nodes() -> None:
    result = _full_result()
    assert result.claims
    assert all(claim["evidence_refs"] for claim in result.claims)
    graph = result.explanation_graph
    claim_nodes = [node for node in graph["nodes"] if node["node_type"] == "claim"]
    assert len(claim_nodes) == len(result.claims)
    assert any(edge["relation"] == "supports_claim" for edge in graph["edges"])
    for payload in result.view_model.human_explanations.values():
        fields = [payload["summary"], *payload["main_reasons"], *payload["model_observed"], *payload["lost_or_averaged"]]
        assert all(re.search(r"\[C-\d{3}\]", text) for text in fields if text)


def test_explanation_levels_and_channel_disclosure() -> None:
    class CallableModel:
        def __call__(self, values):
            return [1 for _ in values]

    partial = FuzzyXAI.wrap(CallableModel()).explain_one([1.0, 0.0], object_id="x")
    assert partial.explanation_level == "E1"
    assert "training_history" in partial.missing_channels
    full = _full_result()
    assert full.explanation_level == "E5"
    assert {"alignment", "reduction", "risk", "counterfactuals"} <= set(full.available_channels)
    assert "rules" in full.native_channels
    assert "rules" not in full.surrogate_channels


def test_result_overview_story_inspect_and_audit() -> None:
    result = _full_result()
    assert "Что решила модель" in result.overview()
    assert "Обучение" in result.story()
    rule = result.inspect("rule:R31")
    assert rule.target["rule_id"] == "R31"
    claim = result.inspect("claim:C001")
    assert claim.target["claim_id"] == "C-001"
    audit = result.audit()
    assert audit["explanation_level"]["level"] == "E5"
    assert audit["trace"]["model_fingerprint"]


def test_visual_spec_separates_metrics_and_renders_views(tmp_path) -> None:
    result = _full_result()
    spec = result.view_model.visual_spec
    profile = spec["data_profile"][0]
    assert profile["reference_interval"][0] is not None
    assert profile["percentile"] is not None
    rule = next(item for item in spec["knowledge_atlas"]["rules"] if item["rule_id"] == "R31")
    assert rule["importance"] == 0.07
    assert rule["coverage"] == 0.08
    assert rule["importance"] != rule["coverage"]
    assert rule["counterfactual_effect"]["test"] == 0.05
    for view in ("explanation_story", "data_profile", "training_trace", "knowledge_atlas", "decision_evidence", "similar_cases", "counterfactual", "rule_ablation", "provenance", "audit"):
        output = result.visualize(view=view, backend="matplotlib", output=tmp_path / f"{view}.png")
        assert output.stat().st_size > 1000
    html = result.visualize(view="explanation_story", backend="plotly", output=tmp_path / "story.html")
    assert html.stat().st_size > 1000
