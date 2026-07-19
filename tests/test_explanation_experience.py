from __future__ import annotations

import re

import pytest

from fuzzyxai import FuzzyXAI
from fuzzyxai.adapters import NativeRuleAdapter
from fuzzyxai.evidence import ExplanationClaim, ExplanationEvidence, evaluate_rule_ablation, extract_rules
from fuzzyxai.schemas import validate_payload
from fuzzyxai.visualization.spec import ExplanationVisualSpec


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
    epochs = (1, 4, 7, 10, 13, 16, 19, 22, 25, 28, 31, 34)
    history = [
        {
            "epoch": epoch,
            "correct": 7 <= epoch < 16,
            "confidence": 0.78 if epoch == 10 else 0.31 if epoch >= 16 else 0.52,
            "loss": 0.24 if epoch == 10 else 0.79 if epoch >= 16 else 0.55,
            "margin": 0.42 if epoch == 10 else -0.16 if epoch >= 16 else 0.05,
            "prototype_distance": 0.35 if epoch < 16 else 0.72,
            "global_metric": 0.78 + 0.006 * index,
            "subgroup_metric": 0.72 if epoch < 16 else 0.49,
            "rule_activations": {"R31": 0.71 if epoch < 16 else 0.08, "R12": 0.55 + 0.02 * index},
        }
        for index, epoch in enumerate(epochs)
    ]
    training = fx.observe_training(
        history={
            "objects": {"85": history},
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
    assert graph.validate_reachability() == ()
    assert {edge.source for edge in graph.edges} <= {node.node_id for node in graph.nodes}
    assert {edge.target for edge in graph.edges} <= {node.node_id for node in graph.nodes}
    assert any(claim.evidence_status == "supported" and claim.effect == "adverse" for claim in result.claims)
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
    assert claim.summary()
    assert claim.evidence()
    assert result.inspect("object:85").target_id == "85"
    assert result.inspect("evidence:data:85").target_type == "evidence"
    assert result.inspect("action").target_id == "action"
    audit = result.audit()
    assert audit["explanation_level"]["level"] == "E5"
    assert audit["trace"]["model_fingerprint"]


def test_claim_contract_rejects_untraceable_or_unbounded_statements() -> None:
    common = {
        "claim_id": "C-X",
        "claim_type": "diagnostic",
        "scope": "object",
        "subject_id": "85",
        "statement": "test",
        "short_statement": "test",
        "effect": "adverse",
        "severity": "warning",
        "strength": 0.5,
    }
    with pytest.raises(ValueError, match="requires at least one evidence"):
        ExplanationClaim(evidence_status="supported", evidence_refs=(), **common)
    with pytest.raises(ValueError, match="fidelity"):
        ExplanationClaim(evidence_status="supported", evidence_refs=("prediction",), surrogate=True, **common)
    with pytest.raises(ValueError, match="research_only"):
        ExplanationClaim(evidence_status="supported", evidence_refs=("prediction",), scope="medical", **{key: value for key, value in common.items() if key != "scope"})


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
    assert len(spec["training_timeline"][0]["points"]) == 12
    assert spec["schema_version"] == "1.1"
    assert spec["audit"]["graph_valid"] is True
    validation = validate_payload(spec, "explanation_visual_spec")
    assert validation.valid, validation.errors
    assert ExplanationVisualSpec.from_dict(spec).to_dict() == spec
    for view in ("explanation_story", "data_profile", "training_trace", "knowledge_atlas", "decision_evidence", "similar_cases", "counterfactual", "rule_ablation", "provenance", "audit"):
        output = result.visualize(view=view, backend="matplotlib", output=tmp_path / f"{view}.png")
        assert output.stat().st_size > 1000
    html = result.visualize(view="explanation_story", backend="plotly", output=tmp_path / "story.html")
    assert html.stat().st_size > 1000
    for view in ("data_profile", "training_trace", "knowledge_atlas", "decision_evidence", "similar_cases", "counterfactual", "rule_ablation", "provenance", "audit"):
        html = result.visualize(view=view, backend="plotly", output=tmp_path / f"{view}.html")
        assert html.stat().st_size > 1000
