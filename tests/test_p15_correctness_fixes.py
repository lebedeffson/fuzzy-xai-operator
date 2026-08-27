"""P15 tier 1: concrete correctness bugs found by an independent review of a
full-data run against real Breast Cancer Wisconsin data (P15.9, P15.10,
P15.11, P15.12 in the review).
"""

from __future__ import annotations

from fuzzyxai import FuzzyXAI
from fuzzyxai.evidence import (
    ExplanationEvidence,
    build_explanation_claims,
    build_explanation_graph,
    build_object_trace,
    compose_human_explanation,
)
from sklearn.datasets import load_breast_cancer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split


def _split():
    X, y = load_breast_cancer(return_X_y=True)
    return train_test_split(X, y, test_size=0.2, random_state=0, stratify=y)


def test_mismatched_label_similar_case_is_marked_counterexample_not_confirmation() -> None:
    X_train, X_test, y_train, _ = _split()
    model = LogisticRegression(max_iter=2000).fit(X_train, y_train)
    train_ids = [f"train_{i}" for i in range(len(X_train))]
    fx = FuzzyXAI.wrap(model, reference_data=X_train, reference_labels=y_train, reference_ids=train_ids)

    found_mismatch = False
    for index in range(20):
        result = fx.explain_one(X_test[index], object_id=f"p{index}")
        predicted = str(result.prediction.predictions[0])
        mismatched = [case for case in result.similar_cases if case["reference_label"] != predicted]
        if not mismatched:
            continue
        found_mismatch = True
        for case in mismatched:
            assert case["is_counterexample"] is True
        human = result.explain_for(audience="domain_user")
        for statement in human.details.similar_cases:
            if any(case["reference_object_id"] in statement.title for case in mismatched):
                lowered = statement.explanation.lower()
                assert "дополнительным подтверждением" not in lowered
                assert "другим результатом" in lowered or "контрпример" in statement.title.lower()
        break
    assert found_mismatch, "test setup did not exercise a mismatched-label neighbor; widen the search"


def test_class_concept_only_favorable_for_predicted_class() -> None:
    X_train, X_test, y_train, _ = _split()
    model = LogisticRegression(max_iter=2000).fit(X_train, y_train)
    fx = FuzzyXAI.wrap(model, reference_data=X_train, reference_labels=y_train)
    result = fx.explain_one(X_test[0], object_id="p0")
    predicted = str(result.prediction.predictions[0])
    concept_claims = [c for c in result.claims if c.claim_type == "class_concept"]
    assert concept_claims
    for claim in concept_claims:
        if claim.subject_id == predicted:
            assert claim.effect == "favorable"
        else:
            assert claim.effect == "neutral"


def test_class_concept_summary_does_not_claim_every_class_supports_result() -> None:
    X_train, X_test, y_train, _ = _split()
    model = LogisticRegression(max_iter=2000).fit(X_train, y_train)
    fx = FuzzyXAI.wrap(model, reference_data=X_train, reference_labels=y_train)
    result = fx.explain_one(X_test[0], object_id="p0")
    predicted = str(result.prediction.predictions[0])
    other_class = "1" if predicted == "0" else "0"
    text = result.summary()
    assert f"группы {other_class}" not in text or "поддерживает результат" not in text.split(f"группы {other_class}")[-1][:80]


def test_counterfactual_plausibility_is_not_fabricated() -> None:
    X_train, X_test, y_train, _ = _split()
    model = LogisticRegression(max_iter=2000).fit(X_train, y_train)
    fx = FuzzyXAI.wrap(model, reference_data=X_train, reference_labels=y_train)
    result = fx.explain_one(X_test[0], object_id="p0", include_counterfactuals=True)
    counterfactual_claims = [c for c in result.claims if c.claim_type == "counterfactual"]
    if not counterfactual_claims:
        return
    human = result.explain_for(audience="domain_user")
    for change in human.what_would_change_result:
        assert change.plausibility is None


def test_counterfactual_wording_distinguishes_classes_without_domain_language() -> None:
    X_train, X_test, y_train, _ = _split()
    model = LogisticRegression(max_iter=2000).fit(X_train, y_train)
    fx = FuzzyXAI.wrap(model, reference_data=X_train, reference_labels=y_train)
    result = fx.explain_one(X_test[0], object_id="p0", include_counterfactuals=True)
    human = result.explain_for(audience="domain_user")
    for change in human.what_would_change_result:
        assert "«другой результат» на «другой результат»" not in change.explanation


def test_observe_training_discloses_ignored_train_val_checkpoint_params() -> None:
    from fuzzyxai.adapters.model import ModelPrediction
    from fuzzyxai.adapters.model_v2 import ModelAdapterV2

    class DummyAdapter(ModelAdapterV2):
        adapter_id = "dummy"
        model_family = "dummy"

        def predict(self, inputs):
            return ModelPrediction(predictions=[0], probabilities=None, model_type="dummy", adapter_id=self.adapter_id, metadata={"task_type": self.task_type.value})

        def feature_names(self):
            return ["a"]

        def model_fingerprint(self):
            return "0" * 16

    fx = FuzzyXAI.wrap(object(), adapter=DummyAdapter(object(), task="classification"))
    analysis = fx.observe_training(train_data=[1, 2, 3], history={"objects": {}})
    assert any("train_data" in item for item in analysis.limitations)

    analysis_clean = fx.observe_training(history={"objects": {}})
    assert analysis_clean.limitations == ()


def test_single_object_forgetting_does_not_overclaim_subgroup_degradation() -> None:
    trace = build_object_trace(
        "obj1",
        [
            {"epoch": 1, "correct": True, "confidence": 0.9},
            {"epoch": 2, "correct": True, "confidence": 0.85},
            {"epoch": 3, "correct": False, "confidence": 0.3},
        ],
    )
    evidence = ExplanationEvidence(training=[trace])
    prediction = {"predictions": [1], "score": 0.7}
    claims = build_explanation_claims(evidence, prediction=prediction, diagnostics=[], action="review")
    graph = build_explanation_graph(evidence, prediction=prediction, diagnostics=[], action="review", claims=claims)
    human = compose_human_explanation(claims, graph, action="review", audience="domain_user", evidence=evidence)
    forgetting_concern = next(c for c in human.concerns if "забы" in c.title.lower() or "объект" in c.title.lower())
    assert "редкие случаи этого типа" not in forgetting_concern.explanation
    assert "редкий тип" not in forgetting_concern.title.lower()


def test_subgroup_degradation_claim_allowed_when_subgroup_evidence_present() -> None:
    from fuzzyxai.evidence import detect_subgroup_averaging

    trace = build_object_trace(
        "obj1",
        [
            {"epoch": 1, "correct": True, "confidence": 0.9},
            {"epoch": 2, "correct": False, "confidence": 0.3},
        ],
    )
    subgroups = detect_subgroup_averaging(
        global_metric=[0.8, 0.85],
        subgroup_metrics={"rare": [0.7, 0.5]},
    )
    evidence = ExplanationEvidence(training=[trace], subgroups=subgroups)
    prediction = {"predictions": [1], "score": 0.7}
    claims = build_explanation_claims(evidence, prediction=prediction, diagnostics=[], action="review")
    if not any(c.claim_type == "forgetting" for c in claims):
        return
    graph = build_explanation_graph(evidence, prediction=prediction, diagnostics=[], action="review", claims=claims)
    human = compose_human_explanation(claims, graph, action="review", audience="domain_user", evidence=evidence)
    assert any("редкий тип" in c.title.lower() for c in human.concerns)


# --- P15.4: linear coefficients are not pseudo-rules -------------------------


def test_linear_model_produces_no_model_rule_claims() -> None:
    X_train, X_test, y_train, _ = _split()
    model = LogisticRegression(max_iter=2000).fit(X_train, y_train)
    fx = FuzzyXAI.wrap(model)
    result = fx.explain_one(X_test[0], object_id="p0")
    assert not [c for c in result.claims if c.claim_type == "model_rule"]


def test_linear_rules_removed_does_not_break_tree_or_forest_rules() -> None:
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.tree import DecisionTreeClassifier

    X_train, X_test, y_train, _ = _split()
    for model in (
        DecisionTreeClassifier(max_depth=3, random_state=0).fit(X_train, y_train),
        RandomForestClassifier(n_estimators=10, max_depth=3, random_state=0).fit(X_train, y_train),
    ):
        fx = FuzzyXAI.wrap(model)
        result = fx.explain_one(X_test[0], object_id="p0")
        assert [c for c in result.claims if c.claim_type == "model_rule"], type(model).__name__


# --- P15.5: capability-aware explanation levels ------------------------------


def test_linear_model_reports_rules_as_not_applicable_not_missing() -> None:
    X_train, X_test, y_train, _ = _split()
    model = LogisticRegression(max_iter=2000).fit(X_train, y_train)
    fx = FuzzyXAI.wrap(model, reference_data=X_train, reference_labels=y_train)
    result = fx.explain_one(X_test[0], object_id="p0")
    level = result.view_model.explanation_level
    assert "rules" in level.get("not_applicable_channels", ())
    assert "rules" not in level["missing_channels"]


def test_tree_model_does_not_report_rules_as_not_applicable() -> None:
    from sklearn.tree import DecisionTreeClassifier

    X_train, X_test, y_train, _ = _split()
    model = DecisionTreeClassifier(max_depth=3, random_state=0).fit(X_train, y_train)
    fx = FuzzyXAI.wrap(model, reference_data=X_train, reference_labels=y_train)
    result = fx.explain_one(X_test[0], object_id="p0")
    level = result.view_model.explanation_level
    assert "rules" not in level.get("not_applicable_channels", ())


def test_not_applicable_defaults_to_empty_when_capability_flags_unspecified() -> None:
    """A caller that doesn't pass native_rules_supported/local_contributions_supported
    gets exactly the pre-P15.5 behavior — nothing reported as not_applicable."""

    from fuzzyxai.evidence import ExplanationEvidence, determine_explanation_level

    level = determine_explanation_level(ExplanationEvidence(), contribution_method=None, operator_channels={})
    assert level.not_applicable_channels == ()


def test_not_applicable_flag_moves_channel_out_of_missing() -> None:
    from fuzzyxai.evidence import ExplanationEvidence, determine_explanation_level

    level = determine_explanation_level(
        ExplanationEvidence(),
        contribution_method=None,
        operator_channels={},
        native_rules_supported=False,
    )
    assert level.not_applicable_channels == ("rules",)
    assert "rules" not in level.missing_channels

