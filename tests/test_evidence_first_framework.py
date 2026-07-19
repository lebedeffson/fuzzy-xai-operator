from __future__ import annotations

import json

import pytest
from sklearn.datasets import make_classification
from sklearn.ensemble import GradientBoostingClassifier, HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier

from fuzzyxai import FuzzyXAI
from fuzzyxai.adapters import NativeRuleAdapter
from fuzzyxai.evidence import (
    build_class_concepts,
    build_object_trace,
    collect_data_evidence,
    evaluate_rule_ablation,
    extract_rules,
    find_similar_tabular_cases,
)
from fuzzyxai.visualization import ExplanationViewModel


class NativeRuleModel:
    classes_ = [0, 1]
    rules_ = [
        {
            "rule_id": "R_high",
            "antecedents": ["water_saturation is high", "distance is low"],
            "consequent": "1",
            "activation": 0.81,
            "coverage": 0.63,
            "precision": 0.88,
            "support": 63,
        }
    ]

    def predict_proba(self, values):
        return [[0.2, 0.8] for _ in values]

    def predict(self, values):
        return [1 for _ in values]


def test_data_evidence_distinguishes_deviation_from_error() -> None:
    reference = [[0.0, 1.0], [0.1, 1.1], [-0.1, 0.9], [0.05, 1.05]]
    item = collect_data_evidence([[4.0, 1.0]], object_ids=["85"], feature_names=["a", "b"], reference_values=reference)[0]
    assert item.object_id == "85"
    assert item.anomaly_labels == ["a"]
    assert "domain validation" in item.warnings[0]


def test_training_trace_detects_real_forgetting_transition() -> None:
    trace = build_object_trace(
        "85",
        [
            {"epoch": 0, "correct": False, "confidence": 0.4},
            {"epoch": 1, "correct": True, "confidence": 0.8},
            {"epoch": 2, "correct": True, "confidence": 0.7},
            {"epoch": 3, "correct": False, "confidence": 0.2},
        ],
    )
    assert trace.first_learned_epoch == 1
    assert trace.forgetting_events == [3]
    assert trace.stability_score < 1.0


@pytest.mark.parametrize(
    "model",
    [
        LogisticRegression(max_iter=500, random_state=42),
        DecisionTreeClassifier(max_depth=3, random_state=42),
        RandomForestClassifier(n_estimators=5, max_depth=3, random_state=42),
        GradientBoostingClassifier(n_estimators=5, max_depth=2, random_state=42),
        HistGradientBoostingClassifier(max_iter=5, random_state=42),
    ],
)
def test_public_api_runs_across_sklearn_model_families(model) -> None:
    values, labels = make_classification(n_samples=90, n_features=4, n_informative=3, n_redundant=0, random_state=42)
    model.fit(values, labels)
    result = FuzzyXAI.wrap(model).explain_one(
        values[0],
        object_id="case-0",
        reference_data=values,
        reference_labels=labels.tolist(),
        feature_names=["a", "b", "c", "d"],
        include_similar_cases=True,
        include_counterfactuals=True,
        include_training_trace=False,
    )
    assert result.prediction.predictions is not None
    assert result.action in {"review", "insufficient_evidence"}
    assert result.view_model.explanation_graph["nodes"]
    assert "## Что делать" in result.summary("user")
    assert result.explain_for().recommended_action.action == result.action


def test_native_anfis_rule_is_not_labelled_surrogate() -> None:
    adapter = NativeRuleAdapter(NativeRuleModel())
    rules = extract_rules(adapter, feature_names=["water_saturation", "distance"], model_version="anfis-test")
    assert len(rules) == 1
    assert rules[0].native is True
    assert rules[0].surrogate is False
    assert rules[0].coverage == 0.63


def test_linear_rule_like_evidence_is_explicitly_surrogate() -> None:
    values, labels = make_classification(n_samples=60, n_features=3, n_informative=2, n_redundant=0, random_state=42)
    model = LogisticRegression(max_iter=500, random_state=42).fit(values, labels)
    rules = extract_rules(FuzzyXAI.wrap(model).model_adapter, feature_names=["a", "b", "c"])
    assert rules
    assert all(rule.surrogate and not rule.native for rule in rules)


def test_rule_ablation_requires_measured_shared_metrics() -> None:
    rule = extract_rules(NativeRuleAdapter(NativeRuleModel()), feature_names=["a", "b"])[0]
    measured = evaluate_rule_ablation(
        rule,
        baseline_metrics={"train": 0.9, "validation": 0.82, "test": 0.80},
        ablated_metrics={"train": 0.88, "validation": 0.74, "test": 0.71},
    )
    assert measured.counterfactual_effect == {"test": 0.09, "train": 0.02, "validation": 0.08}
    assert "measured_rule_ablation" in measured.evidence_refs


def test_similarity_always_names_method_and_representation() -> None:
    cases = find_similar_tabular_cases(
        [0.0, 1.0],
        [[0.1, 1.1], [5.0, -3.0]],
        query_object_id="query",
        feature_names=["shape", "density"],
    )
    assert cases[0].similarity_method == "robust_standardized_euclidean"
    assert cases[0].compared_representation == "normalized tabular feature vector"
    assert cases[0].limitations


def test_class_concept_reports_uncovered_fraction() -> None:
    model = NativeRuleModel()
    rule = extract_rules(NativeRuleAdapter(model), feature_names=["a", "b"])[0]
    concepts = build_class_concepts([[0, 0], [1, 1], [2, 2], [3, 3]], [0, 0, 1, 1], rules=[rule])
    high = next(item for item in concepts if item.class_id == "1")
    assert high.primary_rule_coverage == 0.63
    assert high.uncovered_fraction == pytest.approx(0.37)


def test_serialization_html_dashboard_and_three_text_levels(tmp_path) -> None:
    result = FuzzyXAI.wrap(NativeRuleModel()).explain_one(
        [0.8, 0.2],
        object_id="85",
        reference_data=[[0.7, 0.3], [0.9, 0.1], [0.1, 0.9]],
        reference_labels=[1, 1, 0],
        feature_names=["water_saturation", "distance"],
        include_similar_cases=True,
        include_counterfactuals=True,
        include_training_trace=False,
    )
    json_path = result.export_json(tmp_path / "result.json")
    html_path = result.export_html(tmp_path / "result.html")
    png_path = result.plot(tmp_path / "result.png")
    restored = ExplanationViewModel.load_json(json_path)
    assert restored.to_dict() == result.view_model.to_dict()
    assert html_path.stat().st_size > 0 and png_path.stat().st_size > 0
    assert {"user", "expert", "audit", "domain_user", "ml_engineer", "researcher", "auditor"} == set(
        result.view_model.human_explanations
    )
    assert json.loads(json_path.read_text())["schema_version"] == "2.0"
