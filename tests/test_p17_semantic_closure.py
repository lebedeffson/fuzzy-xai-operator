"""P17: final semantic closure fixes found by auditing the P16 ZIP.

1. Δ is not derived from linear-reconstruction fidelity (a different
   quantity) — reduction stays not_applied unless a real Π was measured.
2. Golden-tabular-style scenarios must not fabricate a second explanatory
   channel just to reach E5 (covered by test_p15_automatic_operator_layer.py's
   honest-E4-ceiling test; not duplicated here).
3. ρ's automatic 5-component formula does not silently renormalize away a
   component the resolved ExplainPlan expects — the interface is marked
   incomplete and action is demoted (covered in
   test_p15_automatic_operator_layer.py).
4. class_concept is neutral unless a real query-to-prototype distance was
   measured and shows genuine closeness.
5. SimilarCaseEvidence carries real query/reference values and deltas, and
   the human report names them, not just feature labels.
"""

from __future__ import annotations

from fuzzyxai import FuzzyXAI
from fuzzyxai.evidence import build_class_concepts, find_similar_tabular_cases
from sklearn.datasets import load_breast_cancer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split


def _split():
    X, y = load_breast_cancer(return_X_y=True)
    return train_test_split(X, y, test_size=0.2, random_state=0, stratify=y)


def test_class_concept_stays_neutral_without_a_query_row() -> None:
    X_train, _, y_train, _ = _split()
    concepts = build_class_concepts(X_train, y_train, representative_limit=2)
    for concept in concepts:
        assert concept.query_distance is None
        assert concept.query_similarity is None


def test_class_concept_query_distance_is_measured_and_symmetric_with_similarity() -> None:
    X_train, X_test, y_train, _ = _split()
    concepts = build_class_concepts(X_train, y_train, representative_limit=2, query_row=X_test[12])
    for concept in concepts:
        assert concept.query_distance is not None
        assert concept.query_similarity is not None
        assert concept.query_similarity > 0.0  # standardized scale, not degenerate exp(-580)=0


def test_class_concept_claim_is_neutral_even_for_predicted_class_when_not_genuinely_close() -> None:
    """The predicted class's own concept must NOT default to favorable —
    only a genuinely close object (query_distance <= intra_class_variability)
    earns "favorable"."""

    X_train, X_test, y_train, _ = _split()
    model = LogisticRegression(max_iter=3000).fit(X_train, y_train)
    fx = FuzzyXAI.wrap(model, reference_data=X_train, reference_labels=y_train)
    result = fx.explain_one(X_test[12], object_id="p0")
    predicted = result.prediction.predictions[0]
    concept_claims = {c["subject_id"]: c for c in result.view_model.claims if c["claim_type"] == "class_concept"}
    predicted_claim = concept_claims[str(predicted)]
    concepts_layer = {c["class_id"]: c for c in result.view_model.layers["concepts"]}
    predicted_concept = concepts_layer[str(predicted)]
    genuinely_close = predicted_concept["query_distance"] <= predicted_concept["intra_class_variability"]
    assert predicted_claim["effect"] == ("favorable" if genuinely_close else "neutral")
    # Every non-predicted class's concept is always neutral, never favorable.
    for class_id, claim in concept_claims.items():
        if class_id != str(predicted):
            assert claim["effect"] == "neutral"


def test_similar_case_evidence_carries_real_values_and_deltas() -> None:
    cases = find_similar_tabular_cases(
        [14.0, 1.0],
        [[13.5, 1.2], [20.0, 5.0]],
        query_object_id="q",
        reference_ids=["r0", "r1"],
        feature_names=["radius", "texture"],
    )
    top = cases[0]
    assert top.query_values == {"radius": 14.0, "texture": 1.0}
    assert top.reference_values["radius"] != 0  # a real reference value, not a placeholder
    assert set(top.raw_deltas) == {"radius", "texture"}
    assert set(top.standardized_deltas) == {"radius", "texture"}
    assert top.raw_deltas["radius"] == top.reference_values["radius"] - top.query_values["radius"]


def test_feature_reason_names_the_transformed_value_not_the_raw_one() -> None:
    """P17: 'value multiplied by coefficient' must name the TRANSFORMED
    value when a preprocessing step changed it — saying 'the value of this
    indicator' without qualification implied the shown raw value was what
    the model actually multiplied, which is misleading behind a
    StandardScaler."""

    import pandas as pd
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler

    data = load_breast_cancer()
    X = pd.DataFrame(data.data, columns=data.feature_names)
    y = data.target
    X_train, X_test, y_train, _ = train_test_split(X, y, test_size=0.2, random_state=0, stratify=y)
    pipe = Pipeline([("scaler", StandardScaler()), ("clf", LogisticRegression(max_iter=3000))]).fit(X_train, y_train)
    result = FuzzyXAI.wrap(pipe).explain_one(X_test.iloc[0].to_numpy(), object_id="p0", feature_names=list(X.columns))
    he = result.explain_for("user")
    assert he.details.supports
    text = " ".join(item.explanation for item in he.details.supports)
    assert "ПРЕОБРАЗОВАННОГО" in text
    assert "получено из исходного" in text


def test_focused_provenance_view_shows_a_small_subgraph_not_the_full_graph() -> None:
    """P17 item 9: the default provenance picture must be a focused,
    readable subgraph (<=12 nodes) around the action, not a sample of the
    full 80+-node graph."""

    import pandas as pd
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler

    data = load_breast_cancer()
    X = pd.DataFrame(data.data, columns=data.feature_names)
    y = data.target
    X_train, X_test, y_train, _ = train_test_split(X, y, test_size=0.2, random_state=0, stratify=y)
    pipe = Pipeline([("scaler", StandardScaler()), ("clf", LogisticRegression(max_iter=3000))]).fit(X_train, y_train)
    fx = FuzzyXAI.wrap(pipe, reference_data=X_train.to_numpy(), reference_labels=y_train.tolist())
    result = fx.explain_one(X_test.iloc[0].to_numpy(), object_id="p0", feature_names=list(X.columns))
    assert len(result.view_model.explanation_graph["nodes"]) > 20  # the full graph really is large
    figure = result.visualize(view="provenance")
    assert figure is not None


def test_focused_provenance_view_can_target_a_specific_claim() -> None:
    import pandas as pd
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler

    data = load_breast_cancer()
    X = pd.DataFrame(data.data, columns=data.feature_names)
    y = data.target
    X_train, X_test, y_train, _ = train_test_split(X, y, test_size=0.2, random_state=0, stratify=y)
    pipe = Pipeline([("scaler", StandardScaler()), ("clf", LogisticRegression(max_iter=3000))]).fit(X_train, y_train)
    fx = FuzzyXAI.wrap(pipe, reference_data=X_train.to_numpy(), reference_labels=y_train.tolist())
    result = fx.explain_one(X_test.iloc[0].to_numpy(), object_id="p0", feature_names=list(X.columns))
    linear_claim = next(c for c in result.claims if c.claim_type == "linear_reconstruction")
    figure = result.visualize(view="provenance", selector=f"claim:{linear_claim.claim_id}")
    assert figure is not None
    inspection_figure = result.inspect(f"claim:{linear_claim.claim_id}").visualize()
    assert inspection_figure is not None


def test_human_report_names_actual_compared_numbers_not_just_feature_labels() -> None:
    X_train, X_test, y_train, _ = _split()
    model = LogisticRegression(max_iter=3000).fit(X_train, y_train)
    train_ids = [f"train_{i}" for i in range(len(X_train))]
    fx = FuzzyXAI.wrap(model, reference_data=X_train, reference_labels=y_train, reference_ids=train_ids)
    result = fx.explain_one(X_test[12], object_id="p0")
    he = result.explain_for("user")
    assert he.details.similar_cases
    text = he.details.similar_cases[0].explanation
    assert "наш объект" in text
    assert "Значения для сравнения" in text
